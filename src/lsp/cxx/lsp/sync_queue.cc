// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "sync_queue.h"

#ifndef CXX_NO_THREADS

namespace cxx::lsp {

auto SyncQueue::closed() -> bool {
  std::unique_lock lock(m_mutex);
  return m_closed;
}

void SyncQueue::close() {
  std::unique_lock lock(m_mutex);
  m_closed = true;
  m_cv.notify_all();
}

void SyncQueue::push(std::function<void()> task) {
  std::unique_lock lock(m_mutex);
  m_queue.push_back(std::move(task));
  m_cv.notify_one();
}

auto SyncQueue::pop() -> std::function<void()> {
  std::unique_lock lock(m_mutex);

  m_cv.wait(lock, [this] { return !m_queue.empty(); });

  auto message = m_queue.front();
  m_queue.pop_front();

  return message;
}

}  // namespace cxx::lsp

#endif