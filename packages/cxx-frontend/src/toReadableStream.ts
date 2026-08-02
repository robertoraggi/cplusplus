// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

type ReadableStreamWithFrom = typeof ReadableStream & {
  from?<T>(iterable: Iterable<T>): ReadableStream<T>;
};

/**
 * Adapts an iterable to a `ReadableStream`.
 *
 * Uses the standard `ReadableStream.from` when the runtime provides it and
 * falls back to a pull based stream otherwise.
 *
 * @param iterable the iterable to adapt.
 * @returns a readable stream of the values of the iterable.
 */
export function toReadableStream<T>(iterable: Iterable<T>): ReadableStream<T> {
  const readableStream = ReadableStream as ReadableStreamWithFrom;

  if (typeof readableStream.from === "function") {
    return readableStream.from(iterable);
  }

  const iterator = iterable[Symbol.iterator]();

  return new ReadableStream<T>({
    pull(controller) {
      const { done, value } = iterator.next();

      if (done) {
        controller.close();
        return;
      }

      controller.enqueue(value);
    },

    cancel(reason) {
      iterator.return?.(reason);
    },
  });
}
