#include "olesnitskiy_v_hoare_sort_simple_merge_seq/omp/include/ops_omp.hpp"

#include <algorithm>
#include <cstddef>
#ifdef _OPENMP
#  include <omp.h>
#endif

#include <stack>
#include <utility>
#include <vector>

namespace olesnitskiy_v_hoare_sort_simple_merge_seq {

OlesnitskiyVHoareSortSimpleMergeOMP::OlesnitskiyVHoareSortSimpleMergeOMP(const std::vector<int> &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

int OlesnitskiyVHoareSortSimpleMergeOMP::HoarePartition(std::vector<int> &array, int left, int right) {
  const int pivot = array[left + ((right - left) / 2)];
  int i = left - 1;
  int j = right + 1;

  while (true) {
    ++i;
    while (array[i] < pivot) {
      ++i;
    }

    --j;
    while (array[j] > pivot) {
      --j;
    }

    if (i >= j) {
      return j;
    }

    std::swap(array[i], array[j]);
  }
}

void OlesnitskiyVHoareSortSimpleMergeOMP::HoareQuickSort(std::vector<int> &array, int left, int right) {
  std::stack<std::pair<int, int>> stack;
  stack.emplace(left, right);

  while (!stack.empty()) {
    auto [current_left, current_right] = stack.top();
    stack.pop();

    if (current_left >= current_right) {
      continue;
    }

    const int middle = HoarePartition(array, current_left, current_right);

    if ((middle - current_left) > (current_right - (middle + 1))) {
      stack.emplace(current_left, middle);
      stack.emplace(middle + 1, current_right);
    } else {
      stack.emplace(middle + 1, current_right);
      stack.emplace(current_left, middle);
    }
  }
}

void OlesnitskiyVHoareSortSimpleMergeOMP::Merge(std::vector<int> &array, int left, int middle, int right) {
  std::vector<int> merged_array;
  const int merged_size = (right - left) + 1;
  merged_array.reserve(static_cast<std::size_t>(merged_size));

  int left_index = left;
  int right_index = middle + 1;

  while (left_index <= middle && right_index <= right) {
    if (array[left_index] <= array[right_index]) {
      merged_array.push_back(array[left_index]);
      ++left_index;
    } else {
      merged_array.push_back(array[right_index]);
      ++right_index;
    }
  }

  while (left_index <= middle) {
    merged_array.push_back(array[left_index]);
    ++left_index;
  }

  while (right_index <= right) {
    merged_array.push_back(array[right_index]);
    ++right_index;
  }

  for (std::size_t index = 0; index < merged_array.size(); ++index) {
    array[static_cast<std::size_t>(left) + index] = merged_array[index];
  }
}

bool OlesnitskiyVHoareSortSimpleMergeOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool OlesnitskiyVHoareSortSimpleMergeOMP::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool OlesnitskiyVHoareSortSimpleMergeOMP::RunImpl() {
  std::vector<int> &array = GetOutput();
  const int size = static_cast<int>(array.size());
  if (size <= 1) {
    return true;
  }

  int chunk_count = 1;
#ifdef _OPENMP
  chunk_count = std::max(1, omp_get_max_threads());
#endif
  chunk_count = std::min(chunk_count, size);

  if (chunk_count == 1) {
    HoareQuickSort(array, 0, size - 1);
    return std::ranges::is_sorted(array);
  }

  std::vector<int> borders(static_cast<std::size_t>(chunk_count + 1));
  for (int border_index = 0; border_index <= chunk_count; ++border_index) {
    borders[static_cast<std::size_t>(border_index)] = (border_index * size) / chunk_count;
  }

#ifdef _OPENMP
#  pragma omp parallel for default(none) shared(array, borders, chunk_count)
#endif
  for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    const int left = borders[static_cast<std::size_t>(chunk_index)];
    const int right = borders[static_cast<std::size_t>(chunk_index) + 1] - 1;
    if (left < right) {
      HoareQuickSort(array, left, right);
    }
  }

  for (int merge_index = 0; merge_index < (chunk_count - 1); ++merge_index) {
    const int middle = borders[static_cast<std::size_t>(merge_index) + 1] - 1;
    const int right = borders[static_cast<std::size_t>(merge_index) + 2] - 1;
    Merge(array, 0, middle, right);
  }

  return std::ranges::is_sorted(array);
}

bool OlesnitskiyVHoareSortSimpleMergeOMP::PostProcessingImpl() {
  return !GetOutput().empty() && std::ranges::is_sorted(GetOutput());
}

}  // namespace olesnitskiy_v_hoare_sort_simple_merge_seq
