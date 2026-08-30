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

#include "async_parse.h"

#include <cxx/preprocessor.h>
#include <cxx/translation_unit.h>

#include <chrono>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace cxx::js {

using emscripten::val;

namespace {

auto cancellationPredicate(std::shared_ptr<bool> cancelled,
                           std::function<bool()> stopParsing)
    -> std::function<bool()> {
  return
      [cancelled = std::move(cancelled), stopParsing = std::move(stopParsing)] {
        if (*cancelled) return true;
        if (!stopParsing) return false;

        *cancelled = stopParsing();
        return *cancelled;
      };
}

auto elapsedMilliseconds(std::chrono::steady_clock::time_point start)
    -> double {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

}  // namespace

auto asyncParse(AsyncParseRequest request) -> val {
  auto unit = request.unit;
  auto exists = request.exists;
  auto readFile = request.readFile;
  auto shouldContinue = request.shouldContinue;
  auto didFinishPhase = std::move(request.didFinishPhase);

  auto cancelled = std::make_shared<bool>(false);

  auto findCandidate =
      [&exists](const std::vector<IncludeCandidate>& candidates)
      -> const IncludeCandidate* {
    if (exists.isUndefined()) return nullptr;

    for (auto& candidate : candidates) {
      if (exists(candidate.fileName).as<bool>()) return &candidate;
    }

    return nullptr;
  };

  auto config = std::move(request.config);
  auto stopParsing = std::move(config.stopParsingPredicate);
  auto shouldStop = cancellationPredicate(cancelled, std::move(stopParsing));
  config.stopParsingPredicate = shouldStop;

  std::optional<PreprocessingState> preprocessingStateSlot;

  const auto preprocessingStartedAt = std::chrono::steady_clock::now();

  unit->beginPreprocessing(std::move(request.source),
                           std::move(request.fileName));

  while (true) {
    if (shouldStop()) break;

    if (!shouldContinue.isUndefined()) {
      val keepGoing = co_await shouldContinue();
      *cancelled = !keepGoing.as<bool>();
      if (shouldStop()) break;
    }

    auto& state = preprocessingStateSlot.emplace(unit->continuePreprocessing());

    if (std::holds_alternative<ProcessingComplete>(state)) break;

    if (auto pendingInclude = std::get_if<PendingInclude>(&state)) {
      auto candidates = pendingInclude->candidates();

      if (auto found = findCandidate(candidates)) {
        pendingInclude->resolveWith(found->fileName, found->isSystemHeader);
      } else {
        pendingInclude->resolveWith(std::nullopt);
      }

    } else if (auto pendingHasIncludes =
                   std::get_if<PendingHasIncludes>(&state)) {
      for (auto& hasInclude : pendingHasIncludes->requests) {
        auto candidates = hasInclude.candidates();
        hasInclude.setExists(findCandidate(candidates) != nullptr);
      }
    } else if (auto pendingFileContent =
                   std::get_if<PendingFileContent>(&state)) {
      if (readFile.isUndefined()) {
        pendingFileContent->setContent(std::nullopt);
        continue;
      }

      val content = co_await readFile(pendingFileContent->fileName);

      if (content.isString()) {
        pendingFileContent->setContent(content.as<std::string>());
      } else {
        pendingFileContent->setContent(std::nullopt);
      }
    }
  }

  unit->endPreprocessing();

  if (didFinishPhase) {
    didFinishPhase("preprocessing", elapsedMilliseconds(preprocessingStartedAt),
                   *cancelled);
  }

  if (*cancelled) co_return val{false};

  const auto parsingStartedAt = std::chrono::steady_clock::now();

  unit->beginParsing(std::move(config));

  while (!*cancelled) {
    auto state = unit->continueParsing();

    if (std::holds_alternative<ParsingComplete>(state)) break;

    if (!shouldContinue.isUndefined()) {
      val keepGoing = co_await shouldContinue();
      *cancelled = !keepGoing.as<bool>();
    }
  }

  unit->endParsing();

  if (didFinishPhase) {
    didFinishPhase("parsing", elapsedMilliseconds(parsingStartedAt),
                   *cancelled);
  }

  co_return val{!*cancelled};
}

}  // namespace cxx::js
