#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "olesnitskiy_v_hoare_sort_simple_merge_seq/common/include/common.hpp"
#include "olesnitskiy_v_hoare_sort_simple_merge_seq/omp/include/ops_omp.hpp"
#include "olesnitskiy_v_hoare_sort_simple_merge_seq/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace olesnitskiy_v_hoare_sort_simple_merge_seq {

class OlesnitskiyVRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param) + "_n" + std::to_string(std::get<0>(test_param).size());
  }

 protected:
  void SetUp() override {
    const TestType &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != input_data_.size()) {
      return false;
    }
    if (!std::ranges::is_sorted(output_data)) {
      return false;
    }

    std::vector<int> expected = input_data_;
    std::ranges::sort(expected);
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(OlesnitskiyVRunFuncTests, HoareSortSimpleMerge) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(std::vector<int>{5}, std::vector<int>{5}, "single_element"),
    std::make_tuple(std::vector<int>{2, 1}, std::vector<int>{1, 2}, "two_elements"),
    std::make_tuple(std::vector<int>{1, 2, 3, 4, 5}, std::vector<int>{1, 2, 3, 4, 5}, "already_sorted"),
    std::make_tuple(std::vector<int>{9, 7, 5, 3, 1}, std::vector<int>{1, 3, 5, 7, 9}, "reverse_sorted"),
    std::make_tuple(std::vector<int>{4, 1, 3, 4, 2, 3, 1}, std::vector<int>{1, 1, 2, 3, 3, 4, 4}, "with_duplicates"),
    std::make_tuple(std::vector<int>{-3, 0, 5, -10, 8, -1}, std::vector<int>{-10, -3, -1, 0, 5, 8}, "negative_values"),
    std::make_tuple(std::vector<int>{10, -5, 0, 10, -5, 8, 2, 2}, std::vector<int>{-5, -5, 0, 2, 2, 8, 10, 10},
                    "mixed_values"),
    std::make_tuple(std::vector<int>{15, 3, 27, 9, 1, 8, 2, 11, 6, 4, 5},
                    std::vector<int>{1, 2, 3, 4, 5, 6, 8, 9, 11, 15, 27}, "odd_size")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<OlesnitskiyVHoareSortSimpleMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_olesnitskiy_v_hoare_sort_simple_merge_seq),
                                           ppc::util::AddFuncTask<OlesnitskiyVHoareSortSimpleMergeOMP, InType>(
                                               kTestParam, PPC_SETTINGS_olesnitskiy_v_hoare_sort_simple_merge_seq));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = OlesnitskiyVRunFuncTests::PrintFuncTestName<OlesnitskiyVRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(HoareSortSimpleMergingTests, OlesnitskiyVRunFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace olesnitskiy_v_hoare_sort_simple_merge_seq
