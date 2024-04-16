#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <vector>
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

   ROOT::RDataFrame f_rdf; 

   std::vector<std::string> fCols;

   std::size_t fChunkSize;
   std::size_t fBatchSize;
   std::size_t fNumChunks;
   std::size_t fMaxChunks;
   std::size_t fNumTrainBatches;
   std::size_t fNumValidationBatches;
   std::size_t fMaxBatches;
   std::size_t fNumColumns;
   std::size_t fNumEntries;
   std::size_t fCurrentRow = 0;
   std::size_t fTrainRemainderRow = 0;
   std::size_t fValRemainderRow = 0;
   std::size_t fTrainRemainder;
   std::size_t fValidationRemainder;
   std::size_t fNumValidation;

   float fValidationSplit;

   std::unique_ptr<TMVA::Experimental::Internal::RBatchLoader<Args...>> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   bool fUseWholeFile = true;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fTrainRemainderTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fValRemainderTensor;   
   
   std::vector<std::vector<std::size_t>> fTrainingIdxs;
   std::vector<std::vector<std::size_t>> fValIndices;

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
        fMaxChunks(maxChunks),
        fNumColumns((numColumns != 0) ? numColumns : cols.size()),
        fShuffle(shuffle),
        fDropRemainder(dropRemainder),
        f_rdf(ROOT::RDataFrame("myTree", "temporary.root")),
        fUseWholeFile(maxChunks == 0)
   {
      // limits the number of batches that can be contained in the batchqueue based on the chunksize
      fMaxBatches = ceil((fChunkSize / fBatchSize) * (1 - fValidationSplit));
      fNumEntries = f_rdf.Count().GetValue();
      fNumChunks = fNumEntries / fChunkSize;

      if (maxChunks != 0 && fNumChunks + 1 > maxChunks){
         fNumChunks = maxChunks;
         fNumEntries = maxChunks * fChunkSize;
      }

      fValidationRemainder = (fNumEntries / fChunkSize) * ceil(fChunkSize * fValidationSplit)
         + ceil((fNumEntries % fChunkSize) * fValidationSplit);
      fTrainRemainder = fNumEntries - fValidationRemainder;
      fNumTrainBatches = fTrainRemainder / fBatchSize;
      fNumValidationBatches = fValidationRemainder / fBatchSize;
      fValidationRemainder %= fBatchSize;
      fTrainRemainder %= fBatchSize;

      // Multiplication and division to avoid floating number error
      fNumValidation = ceil(fChunkSize * fValidationSplit * 100000) / 100000;

      fBatchLoader = std::make_unique<TMVA::Experimental::Internal::RBatchLoader<Args...>>(
         f_rdf, fChunkSize, fBatchSize,fCols, fNumColumns, fMaxBatches, fVecSizes, fVecPadding);

      // Create remainders tensors
      fTrainRemainderTensor =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns});
      fValRemainderTensor =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns});
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

      fCurrentRow = 0;
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

   void LoadChunks()
   {
      for (std::size_t current_chunk = 0; current_chunk < fNumChunks; current_chunk++){
         // stop the loop when the loading is not active anymore
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            if (!fIsActive)
               return;
         }

         fCurrentRow += fChunkSize;
         createIdxs();
         CreateTrainingBatches(current_chunk);
         CreateValidationBatches(current_chunk);
      }

      // Create last chunk which has less than fChunkSize entries
      if (std::size_t leftEntries = fNumEntries % fChunkSize; leftEntries != 0){
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            if (!fIsActive)
               return;
         }
         createIdxs(leftEntries);
         CreateTrainingBatches(fNumChunks);
         CreateValidationBatches(fNumChunks);
      }

      if (!fDropRemainder){
         fBatchLoader->LastBatches(*fTrainRemainderTensor, fTrainRemainderRow, *fValRemainderTensor, fValRemainderRow);
      }

      fBatchLoader->DeActivate();
   }

   void CreateTrainingBatches(std::size_t currentChunk){
      // if (fTrainingIdxs.size() > currentChunk) {
      //    fTrainRemainderRow = fBatchLoader->CreateTrainingBatches(*fTrainRemainderTensor, fTrainRemainderRow, fTrainingIdxs[currentChunk]);
      // } else {
      //    // Create the Validation batches if this is not the first epoch
      //    createIdxs();
      //    fTrainRemainderRow = fBatchLoader->CreateTrainingBatches(*fTrainRemainderTensor, fTrainRemainderRow, fTrainingIdxs[currentChunk]);
      // }
      fTrainRemainderRow = fBatchLoader->CreateTrainingBatches(*fTrainRemainderTensor, fTrainRemainderRow, fTrainingIdxs[currentChunk]);
   }

   void CreateValidationBatches(std::size_t currentChunk){
      fValRemainderRow = fBatchLoader->CreateValidationBatches(*fValRemainderTensor, fValRemainderRow, fValIndices[currentChunk]);
   }

   /// \brief split fChunkSize number of events of the current chunk into validation and training events
   void createIdxs()
   {  
      // Create a vector of number 1..processedEvents
      std::vector<std::size_t> row_order = std::vector<std::size_t>(fChunkSize);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fRng);
      }

      // Devide the vector into training and validation
      std::vector<std::size_t> valid_idx({row_order.begin(), row_order.begin() + fNumValidation});
      std::vector<std::size_t> train_idx({row_order.begin() + fNumValidation, row_order.end()});

      fTrainingIdxs.push_back(train_idx);
      fValIndices.push_back(valid_idx);
   }

      /// @brief split custom number of events of the current chunk into validation and training events
      /// @param processedEvents 
      void createIdxs(std::size_t processedEvents)
   {  
      // Create a vector of number 1..processedEvents
      std::vector<std::size_t> row_order = std::vector<std::size_t>(processedEvents);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fRng);
      }

      // calculate the number of events used for validation
      std::size_t num_validation = ceil(processedEvents * fValidationSplit * 10000) / 10000;

      // Devide the vector into training and validation
      std::vector<std::size_t> valid_idx({row_order.begin(), row_order.begin() + num_validation});
      std::vector<std::size_t> train_idx({row_order.begin() + num_validation, row_order.end()});

      fTrainingIdxs.push_back(train_idx);
      fValIndices.push_back(valid_idx);
   }

   void StartValidation() { fBatchLoader->StartValidation(); }
   bool IsActive() { return fIsActive; }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR
