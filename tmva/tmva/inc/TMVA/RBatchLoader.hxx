#ifndef TMVA_RBatchLoader
#define TMVA_RBatchLoader

#include <vector>
#include <memory>
#include <numeric>

// Imports for threading
#include <queue>
#include <mutex>
#include <condition_variable>

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RLogger.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "TMVA/RTensor.hxx"
#include "TMVA/Tools.h"

namespace TMVA {
namespace Experimental {
namespace Internal {

// RChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename First, typename... Rest>
class RChunkLoaderFunctor {

private:
   std::size_t fOffset = 0;
   std::size_t fVecSizeIdx = 0;
   std::vector<std::size_t> fMaxVecSizes;

   float fVecPadding;

   TMVA::Experimental::RTensor<float> &fChunkTensor;

   /// \brief Load the final given value into fChunkTensor
   /// \tparam First_T
   /// \param first
   template <typename First_T>
   void AssignToTensor(First_T first)
   {
      fChunkTensor.GetData()[fOffset++] = first;
   }

   /// \brief Load the final given value into fChunkTensor
   /// \tparam VecType
   /// \param first
   template <typename VecType>
   void AssignToTensor(const ROOT::RVec<VecType> &first)
   {
      AssignVector(first);
   }

   /// \brief Recursively loop through the given values, and load them onto the fChunkTensor
   /// \tparam First_T
   /// \tparam ...Rest_T
   /// \param first
   /// \param ...rest
   template <typename First_T, typename... Rest_T>
   void AssignToTensor(First_T first, Rest_T... rest)
   {
      fChunkTensor.GetData()[fOffset++] = first;

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   /// \brief Recursively loop through the given values, and load them onto the fChunkTensor
   /// \tparam VecType
   /// \tparam ...Rest_T
   /// \param first
   /// \param ...rest
   template <typename VecType, typename... Rest_T>
   void AssignToTensor(const ROOT::RVec<VecType> &first, Rest_T... rest)
   {
      AssignVector(first);

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   /// \brief Loop through the values of a given vector and load them into the RTensor
   /// Note: the given vec_size does not have to be the same size as the given vector
   ///       If the size is bigger than the given vector, zeros are used as padding.
   ///       If the size is smaller, the remaining values are ignored.
   /// \tparam VecType
   /// \param vec
   template <typename VecType>
   void AssignVector(const ROOT::RVec<VecType> &vec)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();

      for (std::size_t i = 0; i < max_vec_size; i++) {
         if (i < vec_size) {
            fChunkTensor.GetData()[fOffset++] = vec[i];
         } else {
            fChunkTensor.GetData()[fOffset++] = fVecPadding;
         }
      }
   }

public:
   RChunkLoaderFunctor(TMVA::Experimental::RTensor<float> &chunkTensor,
                       const std::vector<std::size_t> &maxVecSizes = std::vector<std::size_t>(),
                       const float vecPadding = 0.0)
      : fChunkTensor(chunkTensor), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding)
   {
   }

   /// \brief Loop through all columns of an event and put their values into an RTensor
   /// \param first
   /// \param ...rest
   void operator()(First first, Rest... rest)
   {
      fVecSizeIdx = 0;
      AssignToTensor(std::forward<First>(first), std::forward<Rest>(rest)...);
   }
};

class RBatchLoader {
private:
   ROOT::RDataFrame & f_rdf;

   std::size_t fChunkSize;
   std::size_t fBatchSize;
   std::vector<std::string> fCols;
   std::size_t fNumColumns;
   std::size_t fMaxBatches;
   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;

   bool fIsActive = false;

   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatches;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   std::size_t fValidationIdx = 0;

   TMVA::Experimental::RTensor<float> fEmptyTensor = TMVA::Experimental::RTensor<float>({0});

public:
   RBatchLoader(ROOT::RDataFrame &rdf, const std::size_t chunkSize, const std::size_t batchSize, const std::vector<std::string> &cols,
                const std::size_t numColumns, const std::size_t maxBatches, const std::vector<std::size_t> &vecSizes = {},
                const float vecPadding = 0.0)
      : f_rdf(rdf),
        fChunkSize(chunkSize),
        fBatchSize(batchSize),
        fCols(cols),
        fNumColumns(numColumns),
        fMaxBatches(maxBatches),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding)
   {
   }

   ~RBatchLoader() { DeActivate(); }

public:
   /// \brief Return a batch of data as a unique pointer.
   /// After the batch has been processed, it should be destroyed.
   /// \return Training batch
   const TMVA::Experimental::RTensor<float> &GetTrainBatch()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      fBatchCondition.wait(lock, [this]() { return !fTrainingBatchQueue.empty() || !fIsActive; });

      if (fTrainingBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fTrainingBatchQueue.front());
      fTrainingBatchQueue.pop();

      fBatchCondition.notify_all();

      return *fCurrentBatch;
   }

   /// \brief Returns a batch of data for validation
   /// The owner of this batch has to be with the RBatchLoader.
   /// This is because the same validation batches should be used in all epochs.
   /// \return Validation batch
   const TMVA::Experimental::RTensor<float> &GetValidationBatch()
   {
      if (HasValidationData()) {
         return *fValidationBatches[fValidationIdx++].get();
      }

      return fEmptyTensor;
   }

   /// \brief Checks if there are more training batches available
   /// \return
   bool HasTrainData()
   {
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         if (!fTrainingBatchQueue.empty() || fIsActive)
            return true;
      }

      return false;
   }

   /// \brief Checks if there are more training batches available
   /// \return
   bool HasValidationData()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      return fValidationIdx < fValidationBatches.size();
   }

   /// \brief Activate the batchloader so it will accept chunks to batch
   void Activate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = true;
      }
      fBatchCondition.notify_all();
   }

   /// \brief DeActivate the batchloader. This means that no more batches are created.
   /// Batches can still be returned if they are already loaded
   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = false;
      }
      fBatchCondition.notify_all();
   }

      /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A size_t scalar defining the number of events processed
   template <typename... Args>
   std::size_t LoadChunk(TMVA::Experimental::RTensor<float> &chunkTensor, const std::size_t currentRow)
   {
      RChunkLoaderFunctor<Args...> func(chunkTensor, fVecSizes, fVecPadding);

      ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, currentRow, currentRow + fChunkSize);

      f_rdf.Foreach(func, fCols);

      return f_rdf.Count().GetValue();
   }

   /// \brief Create a batch filled with the events on the given idx
   /// \param chunkTensor
   /// \param idx
   /// \return
   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateBatch(const TMVA::Experimental::RTensor<float> &chunkTensor, const std::vector<std::size_t> idx)
   {
      auto batch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({fBatchSize, fNumColumns}));

      for (std::size_t i = 0; i < fBatchSize; i++) {
         std::copy(chunkTensor.GetData() + (idx[i] * fNumColumns), chunkTensor.GetData() + ((idx[i] + 1) * fNumColumns),
                   batch->GetData() + i * fNumColumns);
      }

      return batch;
   }

   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateBatch(const TMVA::Experimental::RTensor<float> &chunkTensor, const std::vector<std::size_t> idx, std::size_t batchSize)
   {
      auto batch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({batchSize, fNumColumns}));

      for (std::size_t i = 0; i < batchSize; i++) {
         std::copy(chunkTensor.GetData() + (idx[i] * fNumColumns), chunkTensor.GetData() + ((idx[i] + 1) * fNumColumns),
                   batch->GetData() + i * fNumColumns);
      }

      return batch;
   }

   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateFirstBatch(const TMVA::Experimental::RTensor<float> &chunkTensor,
                  const TMVA::Experimental::RTensor<float> &remainderTensor,
                  std::size_t remainderTensorRow, std::vector<std::size_t> eventIndices){
      auto batch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({fBatchSize, fNumColumns}));
      
      {
         std::vector<std::size_t> idx = std::vector<std::size_t>(remainderTensorRow);
         std::iota(idx.begin(), idx.end(), 0);

         for(size_t i = 0; i < remainderTensorRow; i++){
            std::copy(remainderTensor.GetData() + idx[i] * fNumColumns, remainderTensor.GetData() + (idx[i] + 1) * fNumColumns,
                     batch->GetData() + i * fNumColumns);
         }
      }

      std::vector<std::size_t> idx;
      for (std::size_t i = 0; i < (fBatchSize - remainderTensorRow); i++) {
            idx.push_back(eventIndices[i]);
         }
      
      for(std::size_t i = 0; i < (fBatchSize - remainderTensorRow); i++){
         std::copy(chunkTensor.GetData() + idx[i] * fNumColumns, chunkTensor.GetData() + (idx[i] + 1) * fNumColumns,
                   batch->GetData() + (i + remainderTensorRow) * fNumColumns);
      }

      return batch;
   }

   void SaveRemainingData(const TMVA::Experimental::RTensor<float> &chunkTensor,
                        TMVA::Experimental::RTensor<float> &remainderTensor,
                        const std::size_t remainderTensorRow,
                        std::vector<std::size_t> eventIndices, const std::size_t start){
      std::vector<std::size_t> idx;
      for (std::size_t i = start; i < eventIndices.size(); i++) {
         idx.push_back(eventIndices[i]);
      }

      for (std::size_t i = 0; i < remainderTensorRow; i++){
         std::copy(chunkTensor.GetData() + idx[i] * fNumColumns, chunkTensor.GetData() + (idx[i] + 1) * fNumColumns,
                   remainderTensor.GetData() + i * fNumColumns);
      }
   }

   /// @brief save to remaining data when the whole chunk has to be saved
   /// @param chunkTensor 
   /// @param remainderTensor 
   /// @param remainderTensorRow 
   /// @param eventIndices 
   void SaveRemainingData(const TMVA::Experimental::RTensor<float> &chunkTensor,
                        TMVA::Experimental::RTensor<float> &remainderTensor,
                        const std::size_t remainderTensorRow,
                        std::vector<std::size_t> eventIndices){
      for (std::size_t i = 0; i < eventIndices.size(); i++){
         std::copy(chunkTensor.GetData() + eventIndices[i] * fNumColumns, chunkTensor.GetData() + (eventIndices[i] + 1) * fNumColumns,
                   remainderTensor.GetData() + (i + remainderTensorRow) * fNumColumns);
      }
   }


   /// \brief Create training batches from the given chunk of data based on the given event indices
   /// Batches are added to the training queue of batches
   /// \param chunkTensor
   /// \param eventIndices
   std::size_t CreateTrainingBatches(const TMVA::Experimental::RTensor<float> &chunkTensor,
                              TMVA::Experimental::RTensor<float> &remainderTensor, std::size_t remainderTensorRow,
                              std::vector<std::size_t> eventIndices)
   {
      // Wait until less than a full chunk of batches are in the queue before loading splitting the next chunk into
      // batches
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         fBatchCondition.wait(lock, [this]() { return (fTrainingBatchQueue.size() < fMaxBatches) || !fIsActive; });
         if (!fIsActive)
            return 0;
      }

      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;

      if (eventIndices.size() + remainderTensorRow >= fBatchSize){
         batches.emplace_back(CreateFirstBatch(chunkTensor, remainderTensor, remainderTensorRow, eventIndices));
      }
      else{
         SaveRemainingData(chunkTensor, remainderTensor, remainderTensorRow, eventIndices);
         fBatchCondition.notify_one();
         return remainderTensorRow + eventIndices.size();
      }

      // Create tasks of fBatchSize until all idx are used
      std::size_t start = fBatchSize - remainderTensorRow;
      for (; (start + fBatchSize) <= eventIndices.size(); start += fBatchSize) { //should be less than

         // Grab the first fBatchSize indices from the
         std::vector<std::size_t> idx;
         for (std::size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(eventIndices[i]);
         }

         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, idx));
      }

      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         for (std::size_t i = 0; i < batches.size(); i++) {
            fTrainingBatchQueue.push(std::move(batches[i]));
         }
      }

      fBatchCondition.notify_one();

      remainderTensorRow = eventIndices.size() - start;
      SaveRemainingData(chunkTensor, remainderTensor, remainderTensorRow, eventIndices, start);

      return remainderTensorRow;
   }

   /// \brief Create validation batches from the given chunk based on the given event indices
   /// Batches are added to the vector of validation batches
   /// \param chunkTensor
   /// \param eventIndices
   std::size_t CreateValidationBatches(const TMVA::Experimental::RTensor<float> &chunkTensor,
                                TMVA::Experimental::RTensor<float> &remainderTensor,
                                std::size_t remainderTensorRow,
                                const std::vector<std::size_t> eventIndices)
   {  
      if (eventIndices.size() + remainderTensorRow >= fBatchSize){
         std::unique_ptr<TMVA::Experimental::RTensor<float>> batch = CreateFirstBatch(chunkTensor, remainderTensor, remainderTensorRow, eventIndices);
         std::unique_lock<std::mutex> lock(fBatchLock);
         fValidationBatches.emplace_back(std::move(batch));
      }
      else{
         SaveRemainingData(chunkTensor, remainderTensor, remainderTensorRow, eventIndices);
         fBatchCondition.notify_one();
         return remainderTensorRow + eventIndices.size();
      }

      // Create tasks of fBatchSize untill all idx are used
      std::size_t start = fBatchSize - remainderTensorRow;
      for (; (start + fBatchSize) <= eventIndices.size(); start += fBatchSize) {

         std::vector<std::size_t> idx;

         for (std::size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(eventIndices[i]);
         }

         std::unique_ptr<TMVA::Experimental::RTensor<float>> batch = CreateBatch(chunkTensor, idx);
         std::unique_lock<std::mutex> lock(fBatchLock);
         fValidationBatches.emplace_back(std::move(batch));
      }

      remainderTensorRow = eventIndices.size() - start;
      SaveRemainingData(chunkTensor, remainderTensor, remainderTensorRow, eventIndices, start);

      return remainderTensorRow;
   }

   void LastBatches(const TMVA::Experimental::RTensor<float> &remainderTrainingTensor,
                  const std::size_t remainderTrainingRow,
                  const TMVA::Experimental::RTensor<float> &remainderValidationTensor,
                  const std::size_t remainderValidationRow){
      {  
         std::vector<std::size_t> idx = std::vector<std::size_t>(remainderTrainingRow);
         std::iota(idx.begin(), idx.end(), 0);
         
         std::unique_ptr<TMVA::Experimental::RTensor<float>> batch = CreateBatch(remainderTrainingTensor, idx, remainderTrainingRow);

         std::unique_lock<std::mutex> lock(fBatchLock);
         fTrainingBatchQueue.push(std::move(batch));
      }

      std::vector<std::size_t> idx = std::vector<std::size_t>(remainderValidationRow);
         std::iota(idx.begin(), idx.end(), 0);

      std::unique_ptr<TMVA::Experimental::RTensor<float>> batch = CreateBatch(remainderValidationTensor, idx, remainderValidationRow);
      
      std::unique_lock<std::mutex> lock(fBatchLock);
      fValidationBatches.emplace_back(std::move(batch));
   }

   /// \brief Reset the validation process
   void StartValidation()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      fValidationIdx = 0;
   }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBatchLoader
