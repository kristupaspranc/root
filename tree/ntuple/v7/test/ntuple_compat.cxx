#include "Compression.h"
#include "Rtypes.h"
#include "ntuple_test.hxx"
#include "TKey.h"
#include "ROOT/EExecutionPolicy.hxx"
#include "RXTuple.hxx"
#include <gtest/gtest.h>
#include <memory>
#include <cstdio>

#include "../src/RColumnElement.hxx"

TEST(RNTupleCompat, Epoch)
{
   FileRaii fileGuard("test_ntuple_compat_epoch.root");

   RNTuple ntpl;
   // The first 16 bit integer in the struct is the epoch
   std::uint16_t *versionEpoch = reinterpret_cast<uint16_t *>(&ntpl);
   *versionEpoch = *versionEpoch + 1;
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   file->WriteObject(&ntpl, "ntpl");
   file->Close();

   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   try {
      pageSource->Attach();
      FAIL() << "opening an RNTuple with different epoch version should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported RNTuple epoch version"));
   }
}

TEST(RNTupleCompat, FeatureFlag)
{
   FileRaii fileGuard("test_ntuple_compat_feature_flag.root");

   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetNTuple("ntpl", "");
   descBuilder.SetFeature(RNTupleDescriptor::kFeatureFlagTest);
   descBuilder.AddField(
      RFieldDescriptorBuilder::FromField(ROOT::Experimental::RFieldZero()).FieldId(0).MakeDescriptor().Unwrap());
   ASSERT_TRUE(static_cast<bool>(descBuilder.EnsureValidDescriptor()));

   RNTupleWriteOptions options;
   auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), EContainerFormat::kTFile, options);
   RNTupleSerializer serializer;

   auto ctx = serializer.SerializeHeader(nullptr, descBuilder.GetDescriptor());
   auto buffer = std::make_unique<unsigned char[]>(ctx.GetHeaderSize());
   ctx = serializer.SerializeHeader(buffer.get(), descBuilder.GetDescriptor());
   writer->WriteNTupleHeader(buffer.get(), ctx.GetHeaderSize(), ctx.GetHeaderSize());

   auto szFooter = serializer.SerializeFooter(nullptr, descBuilder.GetDescriptor(), ctx);
   buffer = std::make_unique<unsigned char[]>(szFooter);
   serializer.SerializeFooter(buffer.get(), descBuilder.GetDescriptor(), ctx);
   writer->WriteNTupleFooter(buffer.get(), szFooter, szFooter);

   writer->Commit();
   // Call destructor to flush data to disk
   writer = nullptr;

   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   try {
      pageSource->Attach();
      FAIL() << "opening an RNTuple that uses an unsupported feature should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported format feature: 137"));
   }
}

TEST(RNTupleCompat, FwdCompat_FutureNTupleAnchor)
{
   using ROOT::Experimental::RXTuple;

   constexpr static const char *kNtupleObjName = "ntpl";

   FileRaii fileGuard("test_ntuple_compat_fwd_compat_future_ntuple.root");

   // Write an RXTuple to disk. It is a simulacrum of a future version of RNTuple, with additional fields and a higher
   // class version.
   {
      auto file = std::unique_ptr<TFile>(
         TFile::Open(fileGuard.GetPath().c_str(), "RECREATE", "", ROOT::RCompressionSetting::ELevel::kUncompressed));
      auto xtuple = RXTuple{};
      file->WriteObject(&xtuple, kNtupleObjName);

      // The file is supposed to be small enough to allow for quick scanning by the patching done later.
      // Let's put 4KB as a safe limit.
      EXPECT_LE(file->GetEND(), 4096);
   }

   // Patch all instances of 'RXTuple' -> 'RNTuple'.
   // We do this by just scanning the whole file and replacing all occurrences.
   // This is not the optimal way to go about it, but since the file is small (~1KB)
   // it is fast enough to not matter.
   {
      FILE *f = fopen(fileGuard.GetPath().c_str(), "r+b");

      fseek(f, 0, SEEK_END);
      size_t fsize = ftell(f);

      char *filebuf = new char[fsize];
      fseek(f, 0, SEEK_SET);
      size_t itemsRead = fread(filebuf, fsize, 1, f);
      EXPECT_EQ(itemsRead, 1);

      std::string_view file_view{filebuf, fsize};
      size_t pos = 0;
      while ((pos = file_view.find("XTuple"), pos) != std::string_view::npos) {
         filebuf[pos] = 'N';
         pos += 6; // skip "XTuple"
      }

      fseek(f, 0, SEEK_SET);
      size_t itemsWritten = fwrite(filebuf, fsize, 1, f);
      EXPECT_EQ(itemsWritten, 1);

      fclose(f);
      delete[] filebuf;
   }

   // Read back the RNTuple from the future with TFile
   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      assert(!tfile->IsZombie());
      auto ntuple = std::unique_ptr<RNTuple>(tfile->Get<RNTuple>(kNtupleObjName));
      EXPECT_EQ(ntuple->GetVersionEpoch(), RXTuple{}.fVersionEpoch);
      EXPECT_EQ(ntuple->GetVersionMajor(), RXTuple{}.fVersionMajor);
      EXPECT_EQ(ntuple->GetVersionMinor(), RXTuple{}.fVersionMinor);
      EXPECT_EQ(ntuple->GetVersionPatch(), RXTuple{}.fVersionPatch);
      EXPECT_EQ(ntuple->GetSeekHeader(), RXTuple{}.fSeekHeader);
      EXPECT_EQ(ntuple->GetNBytesHeader(), RXTuple{}.fNBytesHeader);
      EXPECT_EQ(ntuple->GetLenHeader(), RXTuple{}.fLenHeader);
      EXPECT_EQ(ntuple->GetSeekFooter(), RXTuple{}.fSeekFooter);
      EXPECT_EQ(ntuple->GetNBytesFooter(), RXTuple{}.fNBytesFooter);
      EXPECT_EQ(ntuple->GetLenFooter(), RXTuple{}.fLenFooter);
   }

   // Then read it back with RMiniFile
   {
      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple(kNtupleObjName).Unwrap();
      EXPECT_EQ(ntuple.GetVersionEpoch(), RXTuple{}.fVersionEpoch);
      EXPECT_EQ(ntuple.GetVersionMajor(), RXTuple{}.fVersionMajor);
      EXPECT_EQ(ntuple.GetVersionMinor(), RXTuple{}.fVersionMinor);
      EXPECT_EQ(ntuple.GetVersionPatch(), RXTuple{}.fVersionPatch);
      EXPECT_EQ(ntuple.GetSeekHeader(), RXTuple{}.fSeekHeader);
      EXPECT_EQ(ntuple.GetNBytesHeader(), RXTuple{}.fNBytesHeader);
      EXPECT_EQ(ntuple.GetLenHeader(), RXTuple{}.fLenHeader);
      EXPECT_EQ(ntuple.GetSeekFooter(), RXTuple{}.fSeekFooter);
      EXPECT_EQ(ntuple.GetNBytesFooter(), RXTuple{}.fNBytesFooter);
      EXPECT_EQ(ntuple.GetLenFooter(), RXTuple{}.fLenFooter);
   }
}

TEST(RNTupleCompat, NTupleV4)
{
   // A valid RNTuple with ClassVersion 4 and name "myNTuple"
   constexpr const char kNtupleV4Bin[1336] =
      "\x72\x6F\x6F\x74\x00\x00\xF6\xE2\x00\x00\x00\x64\x00\x00\x05\x37\x00\x00\x04\xF4\x00\x00\x00\x43\x00\x00\x00\x01"
      "\x00\x00\x00\x5C\x04\x00\x00\x01\xF9\x00\x00\x03\x51\x00\x00\x01\xA3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x98\x00\x04\x00\x00\x00\x5A\x75\xB6"
      "\xF7\xC4\x00\x3E\x00\x01\x00\x00\x00\x64\x00\x00\x00\x00\x05\x54\x46\x69\x6C\x65\x1C\x74\x65\x73\x74\x5F\x6E\x74"
      "\x75\x70\x6C\x65\x5F\x72\x65\x63\x6F\x6E\x73\x74\x72\x75\x63\x74\x2E\x72\x6F\x6F\x74\x00\x1C\x74\x65\x73\x74\x5F"
      "\x6E\x74\x75\x70\x6C\x65\x5F\x72\x65\x63\x6F\x6E\x73\x74\x72\x75\x63\x74\x2E\x72\x6F\x6F\x74\x00\x00\x05\x75\xB6"
      "\xF7\xC4\x75\xB6\xF7\xC4\x00\x00\x00\x7D\x00\x00\x00\x5C\x00\x00\x00\x64\x00\x00\x00\x00\x00\x00\x02\xD4\x00\x01"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x00\x94\x00\x04\x00\x00\x00\xAC\x75\xB6\xF7\xC4\x00\x22\x00\x01\x00\x00\x00\xFC\x00\x00\x00\x64\x05\x52"
      "\x42\x6C\x6F\x62\x00\x00\x5A\x53\x01\x69\x00\x00\xAC\x00\x00\x28\xB5\x2F\xFD\x20\xAC\x05\x03\x00\x64\x04\x01\x00"
      "\xAC\x00\x08\x00\x00\x00\x6D\x79\x4E\x54\x75\x70\x6C\x65\x0D\x00\x00\x00\x52\x4F\x4F\x54\x20\x76\x36\x2E\x33\x32"
      "\x2E\x30\x32\xC5\xFF\x01\x00\x00\x00\x2F\x00\x02\x00\x00\x00\x70\x74\x05\x00\x00\x00\x66\x6C\x6F\x61\x74\xE0\x14"
      "\x11\x00\x20\xF4\x8A\x45\x01\xC9\xD7\x47\x57\x03\x0A\x00\x2F\x08\x86\xCA\x17\x94\x01\xF9\xBC\x32\xAC\x4E\xA2\x02"
      "\x05\xE3\xC6\xCE\x0C\x28\x07\xB2\x00\x00\x00\x52\x00\x04\x00\x00\x00\x30\x75\xB6\xF7\xC4\x00\x22\x00\x01\x00\x00"
      "\x01\x90\x00\x00\x00\x64\x05\x52\x42\x6C\x6F\x62\x00\x00\x03\x00\x30\x00\x00\x00\x00\x00\x8A\x45\x01\xC9\xD7\x47"
      "\x57\x03\xF4\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xF4\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFE\x7E"
      "\x98\x45\xDD\x66\xDF\xC8\x00\x00\x00\x6C\x00\x04\x00\x00\x00\xAC\x75\xB6\xF7\xC4\x00\x22\x00\x01\x00\x00\x01\xE2"
      "\x00\x00\x00\x64\x05\x52\x42\x6C\x6F\x62\x00\x00\x5A\x53\x01\x41\x00\x00\xAC\x00\x00\x28\xB5\x2F\xFD\x20\xAC\xC5"
      "\x01\x00\x14\x02\x02\x00\xAC\x00\x8A\x45\x01\xC9\xD7\x47\x57\x03\x38\xF4\xFF\xC4\x01\x00\x00\x00\x30\x00\x30\xB2"
      "\x01\x7F\x84\x7E\xE1\xE9\xED\xAD\xBF\x08\x00\x7B\xF0\xB8\x62\x30\x9C\x85\xC0\x95\xBC\x1C\x80\xC3\x78\x66\x76\x38"
      "\x40\x04\x00\x00\x00\x86\x00\x04\x00\x00\x00\x46\x75\xB6\xF7\xC4\x00\x40\x00\x01\x00\x00\x02\x4E\x00\x00\x00\x64"
      "\x1B\x52\x4F\x4F\x54\x3A\x3A\x45\x78\x70\x65\x72\x69\x6D\x65\x6E\x74\x61\x6C\x3A\x3A\x52\x4E\x54\x75\x70\x6C\x65"
      "\x08\x6D\x79\x4E\x54\x75\x70\x6C\x65\x00\x40\x00\x00\x42\x00\x04\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
      "\x00\x00\x01\x1E\x00\x00\x00\x00\x00\x00\x00\x72\x00\x00\x00\x00\x00\x00\x00\xAC\x00\x00\x00\x00\x00\x00\x02\x04"
      "\x00\x00\x00\x00\x00\x00\x00\x4A\x00\x00\x00\x00\x00\x00\x00\xAC\x11\x78\xC1\xFB\x65\x67\x92\xE4\x00\x00\x00\x7D"
      "\x00\x04\x00\x00\x00\x44\x75\xB6\xF7\xC4\x00\x39\x00\x01\x00\x00\x02\xD4\x00\x00\x00\x64\x00\x1C\x74\x65\x73\x74"
      "\x5F\x6E\x74\x75\x70\x6C\x65\x5F\x72\x65\x63\x6F\x6E\x73\x74\x72\x75\x63\x74\x2E\x72\x6F\x6F\x74\x00\x00\x00\x00"
      "\x01\x00\x00\x00\x86\x00\x04\x00\x00\x00\x46\x75\xB6\xF7\xC4\x00\x40\x00\x01\x00\x00\x02\x4E\x00\x00\x00\x64\x1B"
      "\x52\x4F\x4F\x54\x3A\x3A\x45\x78\x70\x65\x72\x69\x6D\x65\x6E\x74\x61\x6C\x3A\x3A\x52\x4E\x54\x75\x70\x6C\x65\x08"
      "\x6D\x79\x4E\x54\x75\x70\x6C\x65\x00\x00\x00\x01\xA3\x00\x04\x00\x00\x04\xF2\x75\xB6\xF7\xC4\x00\x40\x00\x01\x00"
      "\x00\x03\x51\x00\x00\x00\x64\x05\x54\x4C\x69\x73\x74\x0C\x53\x74\x72\x65\x61\x6D\x65\x72\x49\x6E\x66\x6F\x12\x44"
      "\x6F\x75\x62\x6C\x79\x20\x6C\x69\x6E\x6B\x65\x64\x20\x6C\x69\x73\x74\x5A\x4C\x08\x5A\x01\x00\xF2\x04\x00\x78\x01"
      "\xBD\x92\x4D\x4E\xC2\x40\x1C\xC5\x1F\x05\x13\x91\x8F\xAD\x1A\x36\x6E\xBD\x42\x57\x15\x83\x91\x44\x29\x42\xC5\x68"
      "\x82\x66\x80\x29\x94\x8F\x99\x66\xDA\x26\xB2\x63\xE7\x69\xBC\x84\x97\xD0\x53\x78\x05\xFD\x77\x24\x04\x12\x89\x68"
      "\x83\x2F\x99\x69\x3B\xED\xBC\x5F\xFB\x5E\x2D\x64\xDE\xB1\x83\x14\x48\x46\x3C\x91\x52\x16\x32\x6F\x1F\x24\xA7\x19"
      "\x2A\xCE\x26\x5C\x55\x85\x2B\x41\xAB\x2F\xC8\x5A\xC0\x31\x3D\xAE\x37\xA4\x69\x2E\x35\x6C\xDB\x31\xCD\xCA\xA3\xCF"
      "\x95\x37\xE1\x22\x64\x63\xD3\x6C\xD4\x9C\xC8\x1F\x73\xB4\xF3\xCF\x55\x32\xCC\xD0\xD6\x27\x6D\x68\x77\x86\x27\x4A"
      "\xB1\x69\x6C\x16\x21\xBD\xCA\xCD\xC5\x70\xF2\x8F\x56\xD8\x65\x16\x78\x5D\x67\xEA\xF3\xF8\xD6\x1D\x0C\x9A\x9D\xD8"
      "\x11\xA5\xC5\x6B\x00\x28\xB8\x2D\xAE\x02\x4F\x8A\x8A\x2F\xBB\x03\x5A\x40\x9E\x86\x11\x9F\xAC\x53\x31\x12\x81\xD7"
      "\x17\xBC\x77\x14\x0C\xA4\x0A\x2D\xA0\x33\x03\x5E\xE9\xF8\x33\xE5\x92\x0D\xA5\x8A\x8D\xB7\x4B\xF1\xC4\x3F\x50\xEA"
      "\x2C\xFC\x73\x62\x0F\xF3\xC4\x6E\x60\x58\xC0\x95\xEE\xE5\x70\xB9\x97\x9C\xDB\xE4\x7C\x74\xCE\x59\x8F\xEB\xBC\x8A"
      "\x00\x76\x69\xAC\x55\x61\xD1\xCA\x58\x8A\xBE\x05\xB0\xD9\x57\x29\xB7\x30\xE8\xAA\xF9\x5D\xF5\xB5\xF2\x34\xE4\x41"
      "\x12\xC8\xFD\x1C\xD2\xD2\x90\xBA\x86\x1C\x2C\x7F\xC7\x9E\x7B\xC1\x45\x12\xC2\x86\x49\x9D\x49\x19\x6E\x3D\xA9\x24"
      "\x90\xCD\x92\x4A\x42\x68\xCF\xBB\xB8\xD6\x5D\xD8\xBA\x8B\xFD\xE5\x2E\xB2\xEE\xE9\x80\x77\x47\x41\x34\x01\xE9\x97"
      "\x7F\x14\x3E\x01\x15\x3D\xC1\xCA\x00\x00\x00\x43\x00\x04\x00\x00\x00\x0A\x75\xB6\xF7\xC4\x00\x39\x00\x01\x00\x00"
      "\x04\xF4\x00\x00\x00\x64\x00\x1C\x74\x65\x73\x74\x5F\x6E\x74\x75\x70\x6C\x65\x5F\x72\x65\x63\x6F\x6E\x73\x74\x72"
      "\x75\x63\x74\x2E\x72\x6F\x6F\x74\x00\x00\x01\x00\x00\x05\x37\x77\x35\x94\x00";

   FileRaii fileGuard("test_ntuple_compat_ntuplev4.root");
   {
      FILE *f = fopen(fileGuard.GetPath().c_str(), "wb+");
      auto written = fwrite(kNtupleV4Bin, 1, sizeof(kNtupleV4Bin) - 1, f);
      EXPECT_EQ(written, sizeof(kNtupleV4Bin) - 1);
      fclose(f);
   }
   {
      auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
      EXPECT_EQ(reader->GetDescriptor().GetName(), "myNTuple");
   }
   {
      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple("myNTuple").Unwrap();
      EXPECT_EQ(ntuple.GetVersionMajor(), 2);
      EXPECT_EQ(ntuple.GetMaxKeySize(), 0);
   }
}

template <>
class ROOT::Experimental::RField<ROOT::Experimental::Internal::RTestFutureColumn> final
   : public RSimpleField<ROOT::Experimental::Internal::RTestFutureColumn> {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }
   const RColumnRepresentations &GetColumnRepresentations() const final
   {
      static const RColumnRepresentations representations{{{kTestFutureType}}, {}};
      return representations;
   }

public:
   static std::string TypeName() { return "FutureColumn"; }
   explicit RField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

TEST(RNTupleCompat, FutureColumnType)
{
   // Write a RNTuple containing a field with an unknown column type and verify we can
   // read back the ntuple and its descriptor.

   FileRaii fileGuard("test_ntuple_compat_future_col_type.root");
   {
      auto model = RNTupleModel::Create();
      auto col = model->MakeField<ROOT::Experimental::Internal::RTestFutureColumn>("futureColumn");
      auto colValid = model->MakeField<float>("float");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      col->dummy = 0x42424242;
      *colValid = 69.f;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("futureColumn"));
   GTEST_ASSERT_EQ(fdesc.GetLogicalColumnIds().size(), 1);
   const auto &cdesc = desc.GetColumnDescriptor(fdesc.GetLogicalColumnIds()[0]);
   EXPECT_EQ(cdesc.GetType(), EColumnType::kUnknown);

   {
      // Creating a model not in fwd-compatible mode should fail
      EXPECT_THROW(desc.CreateModel(), RException);
   }

   {
      auto modelOpts = RNTupleDescriptor::RCreateModelOptions();
      modelOpts.fForwardCompatible = true;
      auto model = desc.CreateModel(modelOpts);

      // The future column should not show up in the model
      EXPECT_THROW(model->GetConstField("futureColumn"), RException);

      const auto &floatFld = model->GetConstField("float");
      EXPECT_EQ(floatFld.GetTypeName(), "float");

      reader.reset();
      reader = RNTupleReader::Open(std::move(model), "ntpl", fileGuard.GetPath());

      auto floatId = reader->GetDescriptor().FindFieldId("float");
      auto floatPtr = reader->GetView<float>(floatId);
      EXPECT_FLOAT_EQ(floatPtr(0), 69.f);
   }
}

TEST(RNTupleCompat, FutureColumnType_Nested)
{
   // Write a RNTuple containing a field with an unknown column type and verify we can
   // read back the ntuple and its descriptor.

   FileRaii fileGuard("test_ntuple_compat_future_col_type_nested.root");

   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> itemFields;
      itemFields.emplace_back(new RField<std::vector<ROOT::Experimental::Internal::RTestFutureColumn>>("vec"));
      auto field = std::make_unique<ROOT::Experimental::RRecordField>("future", std::move(itemFields));
      model->AddField(std::move(field));
      auto floatP = model->MakeField<float>("float");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *floatP = 33.f;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto futureId = desc.FindFieldId("future");
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("vec._0", futureId));
   GTEST_ASSERT_EQ(fdesc.GetLogicalColumnIds().size(), 1);
   const auto &cdesc = desc.GetColumnDescriptor(fdesc.GetLogicalColumnIds()[0]);
   EXPECT_EQ(cdesc.GetType(), EColumnType::kUnknown);

   {
      // Creating a model not in fwd-compatible mode should fail
      EXPECT_THROW(desc.CreateModel(), RException);
   }

   {
      auto modelOpts = RNTupleDescriptor::RCreateModelOptions();
      modelOpts.fForwardCompatible = true;
      auto model = desc.CreateModel(modelOpts);

      // The future column should not show up in the model
      EXPECT_THROW(model->GetConstField("future"), RException);

      const auto &floatFld = model->GetConstField("float");
      EXPECT_EQ(floatFld.GetTypeName(), "float");

      reader.reset();
      reader = RNTupleReader::Open(std::move(model), "ntpl", fileGuard.GetPath());

      auto floatId = reader->GetDescriptor().FindFieldId("float");
      auto floatPtr = reader->GetView<float>(floatId);
      EXPECT_FLOAT_EQ(floatPtr(0), 33.f);
   }
}

class RFutureField : public RFieldBase {
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RFutureField>(newName);
   };
   void ConstructValue(void *) const final {}

   std::size_t AppendImpl(const void *) final { return 0; }

public:
   RFutureField(std::string_view name)
      : RFieldBase(name, "Future", ROOT::Experimental::Internal::kTestFutureFieldStructure, false)
   {
   }

   std::size_t GetValueSize() const final { return 0; }
   std::size_t GetAlignment() const final { return 0; }
};

TEST(RNTupleCompat, FutureFieldStructuralRole)
{
   // Write a RNTuple containing a field with an unknown structural role and verify we can
   // read back the ntuple, its descriptor and reconstruct the model.

   FileRaii fileGuard("test_ntuple_compat_future_field_struct.root");
   {
      auto model = RNTupleModel::Create();
      auto field = std::make_unique<RFutureField>("future");
      model->AddField(std::move(field));
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("future"));
   EXPECT_EQ(fdesc.GetLogicalColumnIds().size(), 0);

   // Attempting to create a model with default options should fail
   EXPECT_THROW(desc.CreateModel(), RException);

   auto modelOpts = RNTupleDescriptor::RCreateModelOptions();
   modelOpts.fForwardCompatible = true;
   auto model = desc.CreateModel(modelOpts);
   try {
      model->GetConstField("future");
      FAIL() << "trying to get a field with unknown role should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
}

TEST(RNTupleCompat, FutureFieldStructuralRole_Nested)
{
   // Write a RNTuple containing a field with an unknown structural role and verify we can
   // read back the ntuple, its descriptor and reconstruct the model.

   FileRaii fileGuard("test_ntuple_compat_future_field_struct_nested.root");
   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> itemFields;
      itemFields.emplace_back(new RField<int>("int"));
      itemFields.emplace_back(new RFutureField("future"));
      auto field = std::make_unique<ROOT::Experimental::RRecordField>("record", std::move(itemFields));
      model->AddField(std::move(field));
      model->MakeField<float>("float");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("record"));
   EXPECT_EQ(fdesc.GetLogicalColumnIds().size(), 0);

   // Attempting to create a model with default options should fail
   EXPECT_THROW(desc.CreateModel(), RException);

   auto modelOpts = RNTupleDescriptor::RCreateModelOptions();
   modelOpts.fForwardCompatible = true;
   auto model = desc.CreateModel(modelOpts);
   const auto &floatFld = model->GetConstField("float");
   EXPECT_EQ(floatFld.GetTypeName(), "float");
   try {
      model->GetConstField("record");
      FAIL() << "trying to get a field with unknown role should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
}

class RPageSinkTestLocator : public RPageSinkFile {
   ROOT::Experimental::RNTupleLocator WriteSealedPage(const RPageStorage::RSealedPage &sealedPage, std::size_t)
   {
      auto payload = ROOT::Experimental::RNTupleLocatorObject64{0x420};
      RNTupleLocator result;
      result.fPosition = payload;
      result.fType = ROOT::Experimental::Internal::kTestLocatorType;
      result.fBytesOnStorage = sealedPage.GetDataSize();
      return result;
   }

   RNTupleLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) override
   {
      auto element = columnHandle.fColumn->GetElement();
      RPageStorage::RSealedPage sealedPage = SealPage(page, *element);
      return WriteSealedPage(sealedPage, element->GetPackedSize(page.GetNElements()));
   }

public:
   using RPageSinkFile::RPageSinkFile;
};

TEST(RNTupleCompat, UnknownLocatorType)
{
   // Write a RNTuple containing a page with an unknown locator type and verify we can
   // read back the ntuple, its descriptor and reconstruct the model (but not read pages)

   FileRaii fileGuard("test_ntuple_compat_future_locator.root");

   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 14.0);
      auto wopts = RNTupleWriteOptions();
      auto sink = std::make_unique<RPageSinkTestLocator>("ntpl", fileGuard.GetPath(), wopts);
      auto writer = CreateRNTupleWriter(std::move(model), std::move(sink));
      *fieldPt = 33.f;
      writer->Fill();
   }

   auto readOpts = RNTupleReadOptions();
   // disable the cluster cache so we can catch the exception that happens on LoadEntry
   readOpts.SetClusterCache(RNTupleReadOptions::EClusterCache::kOff);
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath(), readOpts);
   const auto &desc = reader->GetDescriptor();
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("pt"));
   EXPECT_EQ(fdesc.GetLogicalColumnIds().size(), 1);

   // Creating a model should succeed
   auto model = desc.CreateModel();
   (void)model;

   try {
      reader->LoadEntry(0);
      FAIL() << "trying to read a field with an unknown locator should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("tried to read a page with an unknown locator"));
   }
}
