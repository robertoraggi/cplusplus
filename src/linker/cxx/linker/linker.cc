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

#include <cxx/linker/layout.h>
#include <cxx/linker/linker.h>
#include <cxx/linker/object.h>
#include <cxx/linker/symbols.h>
#include <cxx/linker/writer.h>

#include <unordered_set>

namespace cxx::linker {

class Linker::Impl {
 public:
  explicit Impl(LinkOptions options) : options_(std::move(options)) {}

  [[nodiscard]] auto addObject(std::string name,
                               std::shared_ptr<const Bytes> content,
                               std::string& error) -> bool {
    auto object = std::make_unique<Object>(std::move(name), *content);
    buffers_.push_back(std::move(content));
    if (!object->parse(error) || !symbols_.add(*object, error)) return false;
    objects_.push_back(std::move(object));
    return true;
  }

  void addArchive(std::shared_ptr<const Archive> archive) {
    archives_.push_back({std::move(archive), {}});
  }

  [[nodiscard]] auto link(Bytes& output, std::string& error) -> bool {
    if (!resolveArchives(error)) return false;
    Layout layout;
    if (!buildLayout(symbols_, objects_, layoutOptions(), layout, error)) {
      return false;
    }
    return writeModule(layout, output, error);
  }

 private:
  struct ArchiveSlot {
    std::shared_ptr<const Archive> archive;
    std::unordered_set<std::uint32_t> extracted;
  };

  LinkOptions options_;
  std::vector<std::shared_ptr<const Bytes>> buffers_;
  std::vector<std::unique_ptr<Object>> objects_;
  std::vector<ArchiveSlot> archives_;
  SymbolTable symbols_;

  [[nodiscard]] auto layoutOptions() const -> LayoutOptions {
    LayoutOptions options;
    options.entry = options_.entry;
    options.exports = options_.exports;
    options.globalBase = options_.globalBase;
    options.stackSize = options_.stackSize;
    options.initialMemory = options_.initialMemory;
    options.maxMemory = options_.maxMemory;
    options.stackFirst = options_.stackFirst;
    options.gcSections = options_.gcSections;
    options.noEntry = options_.noEntry;
    options.stripDebug = options_.stripDebug;
    options.stripAll = options_.stripAll;
    options.allowUndefined = options_.allowUndefined;
    return options;
  }

  [[nodiscard]] auto resolveArchives(std::string& error) -> bool {
    for (auto pending = symbols_.takePending(); !pending.empty();
         pending = symbols_.takePending()) {
      for (const Symbol* symbol : pending) {
        if (!extract(*symbol, error)) return false;
      }
    }
    return true;
  }

  [[nodiscard]] auto extract(const Symbol& symbol, std::string& error) -> bool {
    if (symbol.defined) return true;
    for (auto& slot : archives_) {
      const auto offset = slot.archive->find(symbol.name);
      if (!offset || !slot.extracted.insert(*offset).second) continue;
      auto member = slot.archive->member(*offset);
      if (!member) return true;
      return addMember(std::move(*member), error);
    }
    return true;
  }

  [[nodiscard]] auto addMember(ArchiveMember member, std::string& error)
      -> bool {
    auto object =
        std::make_unique<Object>(std::move(member.name), member.content);
    if (!object->parse(error) || !symbols_.add(*object, error)) return false;
    objects_.push_back(std::move(object));
    return true;
  }
};

Linker::Linker(LinkOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

Linker::~Linker() = default;

auto Linker::addObject(std::string name, std::shared_ptr<const Bytes> content,
                       std::string& error) -> bool {
  return impl_->addObject(std::move(name), std::move(content), error);
}

void Linker::addArchive(std::shared_ptr<const Archive> archive) {
  impl_->addArchive(std::move(archive));
}

auto Linker::link(Bytes& output, std::string& error) -> bool {
  return impl_->link(output, error);
}

}  // namespace cxx::linker
