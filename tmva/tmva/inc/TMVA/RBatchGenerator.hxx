#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <vector>
#include <functional>
#include <thread>
#include <memory>
#include <cmath>
#include <mutex>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "TMVA/RBatchLoader.hxx"
#include "TMVA/Tools.h"
#include "TRandom3.h"
#include "TROOT.h"

namespace TMVA {
namespace Experimental {
namespace Internal {

template <typename... Args>
class RBatchGenerator {
private:
   TMVA::RandomGenerator<TRandom3> fRng = TMVA::RandomGenerator<TRandom3>(0);
   UInt_t fFixedSeed;
   TMVA::RandomGenerator<TRandom3> fFixedRng;

   ROOT::RDataFrame f_rdf; 

   std::vector<std::string> fCols;

   std::size_t fChunkSize;
   std::size_t fBatchSize;
   std::size_t fNumChunks;
   std::size_t fMaxBatches;
   std::size_t fNumColumns;
   std::size_t fNumEntries;
   std::size_t fTrainRemainderRow = 0;
   std::size_t fValRemainderRow = 0;
   std::size_t fNumValidation;
   std::size_t fUnfilledChunk;

   float fValidationSplit;

   std::unique_ptr<TMVA::Experimental::Internal::RBatchLoader<Args...>> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fTrainRemainderTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fValRemainderTensor;   
   
   std::vector<std::size_t> fTrainIndices;
   std::vector<std::size_t> fValIndices;

   // filled batch elements
   std::mutex fIsActiveLock;

   bool fDropRemainder = true;
   bool fShuffle = true;
   bool fIsActive = false;

   std::vector<std::size_t> fVecSizes;
   float fVecPadding;

public:
   RBatchGenerator(/*ROOT::RDF::RNode &rdf,*/ const std::size_t chunkSize,
                   const std::size_t batchSize, const std::vector<std::string> &cols,
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                   const float validationSplit = 0.0, const std::size_t maxChunks = 0, const std::size_t numColumns = 0,
                   bool shuffle = true, bool dropRemainder = true)
      : fChunkSize(chunkSize),
        fBatchSize(batchSize),
        fCols(cols),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fValidationSplit(validationSplit),
        fNumColumns((numColumns != 0) ? numColumns : cols.size()),
        fShuffle(shuffle),
        fDropRemainder(dropRemainder),
        f_rdf(ROOT::RDataFrame("myTree", "temporary.root"))
   {
      // limits the number of batches that can be contained in the batchqueue based on the chunksize
      fMaxBatches = ceil((fChunkSize / fBatchSize) * (1 - fValidationSplit));
      fNumEntries = f_rdf.Count().GetValue();
      fNumChunks = fNumEntries / fChunkSize;

      if (maxChunks != 0 && fNumChunks + 1 > maxChunks){
         fNumChunks = maxChunks;
         fNumEntries = maxChunks * fChunkSize;
      }

      fUnfilledChunk = fNumEntries % fChunkSize;

      // Multiplication and division to avoid floating number error
      fNumValidation = ceil(fChunkSize * fValidationSplit * 1000000) / 1000000;

      {
         std::function<UInt_t(UInt_t)> GetSeedNumber;
         GetSeedNumber = [&](UInt_t seed_number)->UInt_t{return seed_number != 0? seed_number: GetSeedNumber(fRng());};
         fFixedSeed = GetSeedNumber(fRng());
      }
      
      fFixedRng = TMVA::RandomGenerator<TRandom3>(fFixedSeed);

      fBatchLoader = std::make_unique<TMVA::Experimental::Internal::RBatchLoader<Args...>>(
         f_rdf, fChunkSize, fBatchSize,fCols, fNumColumns, fMaxBatches, fVecSizes, fVecPadding);

      // Create remainders tensors
      fTrainRemainderTensor =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
      fValRemainderTensor =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
   }

   ~RBatchGenerator() { DeActivate(); }

   /// \brief De-activate the loading process by deactivating the batchgenerator
   /// and joining the loading thread
   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fIsActiveLock);
         fIsActive = false;
      }

      fBatchLoader->DeActivate();

      if (fLoadingThread) {
         if (fLoadingThread->joinable()) {
            fLoadingThread->join();
         }
      }
   }

   /// \brief Activate the loading process by starting the batchloader, and
   /// spawning the loading thread.
   void Activate()
   {
      if (fIsActive)
         return;

      {
         std::lock_guard<std::mutex> lock(fIsActiveLock);
         fIsActive = true;
      }

      fFixedRng.seed(fFixedSeed);
      fBatchLoader->Activate();
      fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   /// \brief Returns the next batch of training data if available.
   /// Returns empty RTensor otherwise.
   /// \return
   const TMVA::Experimental::RTensor<float> &GetTrainBatch()
   {
      // Get next batch if available
      return fBatchLoader->GetTrainBatch();
   }

   /// \brief Returns the next batch of validation data if available.
   /// Returns empty RTensor otherwise.
   /// \return
   const TMVA::Experimental::RTensor<float> &GetValidationBatch()
   {
      // Get next batch if available
      return fBatchLoader->GetValidationBatch();
   }

   bool HasTrainData() { return fBatchLoader->HasTrainData(); }

   bool HasValidationData() { return fBatchLoader->HasValidationData(); }

   std::size_t NumberOfTrainingBatches(){
      if (fDropRemainder || !fUnfilledChunk){
         return ((fNumEntries / fChunkSize) * (fChunkSize - fNumValidation) + floor((fNumEntries % fChunkSize) * (1 - fValidationSplit) * 1000000)/1000000) / fBatchSize;
      }

      return ((fNumEntries / fChunkSize) * (fChunkSize - fNumValidation) + floor((fNumEntries % fChunkSize) * (1 - fValidationSplit) * 1000000)/1000000) / fBatchSize;
   }

   std::size_t NumberOfValidationBatches(){
      if (fUnfilledChunk == ceil(fUnfilledChunk * fValidationSplit * 1000000) / 1000000 && (fDropRemainder || !fUnfilledChunk)){
         return ((fNumEntries / fChunkSize) * fNumValidation + ceil((fNumEntries % fChunkSize) * fValidationSplit * 1000000)/1000000) / fBatchSize;
      }
      
      return ((fNumEntries / fChunkSize) * fNumValidation + ceil((fNumEntries % fChunkSize) * fValidationSplit * 1000000)/1000000) / fBatchSize + 1;
   }

   void LoadChunks()
   {
      for (std::size_t current_chunk = 0; current_chunk < fNumChunks; current_chunk++){
         // stop the loop when the loading is not active anymore
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            if (!fIsActive)
               return;
         }

         createIndices();
         fTrainRemainderRow = fBatchLoader->CreateTrainingBatches(*fTrainRemainderTensor, fTrainRemainderRow, fTrainIndices);
         fValRemainderRow = fBatchLoader->CreateValidationBatches(*fValRemainderTensor, fValRemainderRow, fValIndices);
      }

      // Create last chunk which has less than fChunkSize entries
      if (fUnfilledChunk){
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            if (!fIsActive)
               return;
         }

         createIndices(fUnfilledChunk);
         fTrainRemainderRow = fBatchLoader->CreateTrainingBatches(*fTrainRemainderTensor, fTrainRemainderRow, fTrainIndices);
         fValRemainderRow = fBatchLoader->CreateValidationBatches(*fValRemainderTensor, fValRemainderRow, fValIndices);
      }

      if (!fDropRemainder){
         fBatchLoader->LastBatches(*fTrainRemainderTensor, fTrainRemainderRow, *fValRemainderTensor, fValRemainderRow);
      }

      fBatchLoader->DeActivate();
   }

   /// \brief split fChunkSize number of events of the current chunk into validation and training events
   void createIndices()
   {  
      // Create a vector of number 1..processedEvents
      std::vector<std::size_t> row_order = std::vector<std::size_t>(fChunkSize);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fFixedRng);
      }

      // Devide the vector into training and validation
      fTrainIndices = std::vector<std::size_t>{row_order.begin(), row_order.end() - fNumValidation};
      fValIndices = std::vector<std::size_t>{row_order.end() - fNumValidation, row_order.end()};

      if (fShuffle) {
         std::shuffle(fTrainIndices.begin(), fTrainIndices.end(), fRng);
      }
   }

      /// @brief split custom number of events of the current chunk into validation and training events
      /// @param processedEvents 
      void createIndices(std::size_t processedEvents)
   {  
      // Create a vector of number 1..processedEvents
      std::vector<std::size_t> row_order = std::vector<std::size_t>(processedEvents);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fFixedRng);
      }

      // calculate the number of events used for validation
      std::size_t num_validation = ceil(processedEvents * fValidationSplit * 1000000) / 1000000;

      // Devide the vector into training and validation
      fTrainIndices = std::vector<std::size_t>{row_order.begin(), row_order.end() - num_validation};
      fValIndices = std::vector<std::size_t>{row_order.end() - num_validation, row_order.end()};

      if (fShuffle) {
         std::shuffle(fTrainIndices.begin(), fTrainIndices.end(), fRng);
      }
   }

   void StartValidation() { fBatchLoader->StartValidation(); }
   bool IsActive() { return fIsActive; }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR
