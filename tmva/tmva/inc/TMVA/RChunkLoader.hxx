#ifndef TMVA_CHUNKLOADER
#define TMVA_CHUNKLOADER

#include <vector>
#include <iostream>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

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

   std::shared_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;

   /// \brief Load the final given value into fChunkTensor
   /// \tparam First_T
   /// \param first
   template <typename First_T>
   void AssignToTensor(First_T first)
   {  
      std::cout << first << "\n";
      fChunkTensor->GetData()[fOffset++] = first;
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
      std::cout << first << "\n";
      fChunkTensor->GetData()[fOffset++] = first;

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
            fChunkTensor->GetData()[fOffset++] = vec[i];
         } else {
            fChunkTensor->GetData()[fOffset++] = fVecPadding;
         }
      }
   }

public:
   RChunkLoaderFunctor(std::shared_ptr<TMVA::Experimental::RTensor<float>> chunkTensor,
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


template <typename First, typename... Rest>
class RChunkLoaderFunctorFilters {

private:
   std::size_t fOffset = 0;
   std::size_t fVecSizeIdx = 0;
   std::vector<std::size_t> fMaxVecSizes;
   std::size_t fEntries = 0;
   std::size_t fChunkSize;

   float fVecPadding;

   std::shared_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::shared_ptr<TMVA::Experimental::RTensor<float>> fRemainderTensor;

   /// \brief Load the final given value into fChunkTensor
   /// \tparam First_T
   /// \param first
   template <typename First_T>
   void AssignToTensor(First_T first)
   {  
      std::cout << first << " In last valueand here is the offset "<< fOffset << "\n";;
      fChunkTensor->GetData()[fOffset++] = first;
      fEntries++;
      if(fEntries == fChunkSize){
         std::cout << "Entries equal to chunksize\n";
         fChunkTensor = fRemainderTensor;
         fOffset = 0;
         }
      std::cout << "Offset " << fOffset << "\n";
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
      std::cout << first << " In the middle of the process and here is the offset "<< fOffset << "\n";
      fChunkTensor->GetData()[fOffset++] = first;

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
            fChunkTensor->GetData()[fOffset++] = vec[i];
         } else {
            fChunkTensor->GetData()[fOffset++] = fVecPadding;
         }
      }
   }

public:
   RChunkLoaderFunctorFilters(std::shared_ptr<TMVA::Experimental::RTensor<float>> chunkTensor,
                       std::shared_ptr<TMVA::Experimental::RTensor<float>> remainderTensor,
                       std::size_t entries, std::size_t chunkSize, std::size_t && offset,
                       const std::vector<std::size_t> &maxVecSizes = std::vector<std::size_t>(),
                       const float vecPadding = 0.0)
      : fChunkTensor(chunkTensor), fRemainderTensor(remainderTensor), fEntries(entries),
        fChunkSize(chunkSize), fOffset(offset), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding)
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

   std::size_t & SetEntries(){ return fEntries; }
   std::size_t & SetOffset(){ return fOffset; }
};

template <typename... Args>
class RChunkLoader {

private:
   std::size_t fChunkSize;
   std::size_t fNumColumns;

   std::vector<std::string> fCols;

   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;

   ROOT::RDF::RNode & f_rdf;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param rdf
   /// \param chunkSize
   /// \param cols
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoader(ROOT::RDF::RNode &rdf, const std::size_t chunkSize,
                const std::vector<std::string> &cols,
                const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0)
      : f_rdf(rdf),
        fChunkSize(chunkSize),
        fCols(cols),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fNumColumns(cols.size())
   {
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::pair<std::size_t, std::size_t>
   LoadChunk(std::shared_ptr<TMVA::Experimental::RTensor<float>> chunkTensor, const std::size_t currentRow)
   {  
      std::cout << "Me with no filters running!\n";
      RChunkLoaderFunctor<Args...> func(chunkTensor, fVecSizes, fVecPadding);

      ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, currentRow, currentRow + fChunkSize);
      auto myCount = f_rdf.Count();

      // load data
      f_rdf.Foreach(func, fCols);

      // get loading info
      std::size_t processed_events = myCount.GetValue();
      std::size_t passed_events = myCount.GetValue();
      return std::make_pair(processed_events, passed_events);
   }
};

template <typename... Args>
class RChunkLoaderFilters {

private:
   ROOT::RDF::RNode & f_rdf;
   std::size_t fChunkSize;
   std::vector<std::string> fCols;
   const std::size_t fNumEntries;
   std::size_t fNumAllEntries;
   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;
   std::size_t fNumColumns;

   const std::size_t fPartOfChunkSize;
   std::shared_ptr<TMVA::Experimental::RTensor<float>> fRemainderChunkTensor;
   std::size_t fRemainderChunkTensorRow = 0;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param rdf
   /// \param chunkSize
   /// \param cols
   /// \param filters
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoaderFilters(ROOT::RDF::RNode &rdf, const std::size_t chunkSize,
                const std::vector<std::string> &cols, std::size_t numEntries,
                std::size_t numAllEntries, const std::vector<std::size_t> &vecSizes = {},
                const float vecPadding = 0.0)
      : f_rdf(rdf),
        fChunkSize(chunkSize),
        fCols(cols),
        fNumEntries(numEntries),
        fNumAllEntries(numAllEntries),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fNumColumns(cols.size()),
        fPartOfChunkSize(chunkSize/5),
        fRemainderChunkTensor(
            std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fPartOfChunkSize, fNumColumns}))
   {
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::pair<std::size_t, std::size_t>
   LoadChunk(std::shared_ptr<TMVA::Experimental::RTensor<float>> chunkTensor, std::size_t currentRow)
   {  
      std::cout << "Part chunk size: " << fPartOfChunkSize << "\n";
      std::cout << "Remainder chunk tensor row: " << fRemainderChunkTensorRow << "\n";
      for (std::size_t i = 0; i < fRemainderChunkTensorRow; i++){
         std::copy(fRemainderChunkTensor->GetData() + (i*fNumColumns), fRemainderChunkTensor->GetData() + ((i+1)*fNumColumns),
                  chunkTensor->GetData() + (i*fNumColumns));
      }

      RChunkLoaderFunctorFilters<Args...> func(chunkTensor, fRemainderChunkTensor, fRemainderChunkTensorRow,
         fChunkSize, fRemainderChunkTensorRow * fNumColumns, fVecSizes, fVecPadding);

      std::size_t passedEvents = 0;
      std::size_t processedEvents = 0;
      std::cout << "fNumAllEntries: " << fNumAllEntries << "\n";
      while((passedEvents < fChunkSize && passedEvents < fNumEntries) && currentRow < fNumAllEntries){
         std::cout << "Passed events: " << passedEvents << "\n";
         std::cout << "Current row: " << currentRow << "\n";

         ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, currentRow, currentRow + fPartOfChunkSize);
         auto report = f_rdf.Report();

         f_rdf.Foreach(func, fCols);
         
         // std::cout << "Slice of tensor\n";
         // for (std::size_t a = currentRow * 2; a < (currentRow + fPartOfChunkSize)*2; a++){
         //    std::copy(chunkTensor->GetData() + a, chunkTensor->GetData() + a + 1, std::ostream_iterator<int>(std::cout, ", "));
         // }

         processedEvents += report.begin()->GetAll();
         passedEvents += (report.end() - 1)->GetPass();

         currentRow += fPartOfChunkSize;
         func.SetEntries() = passedEvents;
         func.SetOffset() = passedEvents * fNumColumns;
      }

      std::cout << "Full inside LoadChunk\n";
      for (std::size_t a = 0; a < passedEvents*2; a++){
         std::copy(chunkTensor->GetData() + a, chunkTensor->GetData() + a + 1, std::ostream_iterator<int>(std::cout, ", "));
      }

      std::cout << "Passed events: " << passedEvents << "\n";
      std::cout << "Current row: " << currentRow << "\n";

      fRemainderChunkTensorRow = passedEvents > fChunkSize? passedEvents - fChunkSize: 0;

      return std::make_pair(processedEvents, passedEvents);
   }

   std::size_t LastChunk(std::shared_ptr<TMVA::Experimental::RTensor<float>> chunkTensor){
      for (std::size_t i = 0; i < fRemainderChunkTensorRow; i++){
         std::copy(fRemainderChunkTensor->GetData() + (i*fNumColumns), fRemainderChunkTensor->GetData() + ((i+1)*fNumColumns),
                  chunkTensor->GetData() + (i*fNumColumns));
      }

      return fRemainderChunkTensorRow;
   }
};
} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_CHUNKLOADER
