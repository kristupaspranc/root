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
   std::size_t fTrainingRemainderRow = 0;
   std::size_t fValidationRemainderRow = 0;
   std::size_t fTrainRemainder;
   std::size_t fValidationRemainder;

   float fValidationSplit;

   std::unique_ptr<TMVA::Experimental::Internal::RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   bool fUseWholeFile = true;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fTrainRemainderTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fValidationRemainderTensor;   

   std::vector<std::vector<std::size_t>> fTrainingIdxs;
   std::vector<std::vector<std::size_t>> fValidationIdxs;

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
      fNumChunks = fNumEntries % fChunkSize != 0? fNumEntries / fChunkSize + 1: fNumEntries / fChunkSize;

      if (maxChunks != 0 && fNumChunks > maxChunks){
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

      fBatchLoader = std::make_unique<TMVA::Experimental::Internal::RBatchLoader>(
         f_rdf, fChunkSize, fBatchSize,fCols, fNumColumns, fMaxBatches, fVecSizes, fVecPadding);

      // Create tensor to load the chunk into
      fChunkTensor =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fChunkSize, fNumColumns});
      // Create remainders tensors
      fTrainRemainderTensor =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
      fTrainRemainderTensor =
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
      for (std::size_t current_chunk = 0; current_chunk < fNumChunks;current_chunk++){
         // stop the loop when the loading is not active anymore
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            if (!fIsActive)
               return;
         }

         // A pair that consists the proccessed, and passed events while loading the chunk
         std::size_t report = fBatchLoader->template LoadChunk <Args...>(*fChunkTensor, fCurrentRow);
         fCurrentRow += report;

         CreateBatches(current_chunk, report);
      }

      if (!fDropRemainder){
         fBatchLoader->LastBatches(*fTrainRemainderTensor, fTrainingRemainderRow, *fTrainRemainderTensor, fValidationRemainderRow);
      }

      fBatchLoader->DeActivate();
   }

   /// \brief Create batches for the current_chunk.
   /// \param currentChunk
   /// \param processedEvents
   void CreateBatches(std::size_t currentChunk, std::size_t processedEvents)
   {
      // Check if the indices in this chunk where already split in train and validations
      if (fTrainingIdxs.size() > currentChunk) {
         fTrainingRemainderRow = fBatchLoader->CreateTrainingBatches(*fChunkTensor, *fTrainRemainderTensor, fTrainingRemainderRow, fTrainingIdxs[currentChunk]);
      } else {
         // Create the Validation batches if this is not the first epoch
         createIdxs(processedEvents);
         fTrainingRemainderRow = fBatchLoader->CreateTrainingBatches(*fChunkTensor, *fTrainRemainderTensor, fTrainingRemainderRow, fTrainingIdxs[currentChunk]);
         fValidationRemainderRow = fBatchLoader->CreateValidationBatches(*fChunkTensor, *fTrainRemainderTensor, fValidationRemainderRow, fValidationIdxs[currentChunk]);
      }
   }

   /// \brief plit the events of the current chunk into validation and training events
   /// \param processedEvents
   void createIdxs(std::size_t processedEvents)
   {  
      // Create a vector of number 1..processedEvents
      std::vector<std::size_t> row_order = std::vector<std::size_t>(processedEvents);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fRng);
      }

      // calculate the number of events used for validation
      std::size_t num_validation = ceil(processedEvents * fValidationSplit);

      // Devide the vector into training and validation
      std::vector<std::size_t> valid_idx({row_order.begin(), row_order.begin() + num_validation});
      std::vector<std::size_t> train_idx({row_order.begin() + num_validation, row_order.end()});

      fTrainingIdxs.push_back(train_idx);
      fValidationIdxs.push_back(valid_idx);
   }

   void StartValidation() { fBatchLoader->StartValidation(); }
   bool IsActive() { return fIsActive; }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR
