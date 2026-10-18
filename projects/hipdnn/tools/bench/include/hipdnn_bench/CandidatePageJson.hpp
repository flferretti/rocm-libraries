// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/EngineQueryTypes.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

/// @file CandidatePageJson.hpp
/// @brief How the bench names a graph, and the JSON a catalog enumeration or sweep prints.
namespace hipdnn_bench
{

/// @brief The content key of a serialized graph (or of the graph inside a graph-and-plan
/// container): `GraphContentKey::hash()` in decimal, reported as `graph_key`.
///
/// The key ignores the graph's name and id and numbers tensors by position, so two graphs
/// that compute the same thing share it. It is what a measurement is stored, deduplicated
/// and held out under. hipdnn_corpus_gen's `graphContentKey` computes the same value, so a
/// corpus manifest's `benchmark` is the key the bench reports for that file.
inline std::string graphContentKey(const uint8_t* data, size_t size)
{
    const auto graph
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(data,
                                                                                         size);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey key{graph};
    if(!key.isUsable())
    {
        throw std::runtime_error("graph has no verifiable serialized content to key");
    }
    return std::to_string(key.hash());
}

/// A knob tuple as a JSON object of knob name to value.
inline nlohmann::json
    knobJson(const std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant>& knobs)
{
    nlohmann::json result = nlohmann::json::object();
    for(const auto& entry : knobs)
    {
        std::visit([&result, &entry](const auto& value) { result[entry.first] = value; },
                   entry.second);
    }
    return result;
}

/// One slot's kernel inside a slotted candidate (RFC 0021 §5).
struct SlotKernel
{
    std::string slot;
    std::string kernelId;
};

/// A slotted engine's composite candidate id, `prep=<id>|dkdv=<id>|dq=<id>`, split into its
/// slots in id order. The id of an unslotted candidate is a bare kernel id (a UUID carries no
/// `=`), which yields no slots. An id that starts like a composite and is malformed throws.
inline std::vector<SlotKernel> slotKernels(const std::string& candidateId)
{
    std::vector<SlotKernel> kernels;
    if(candidateId.find('=') == std::string::npos)
    {
        return kernels;
    }
    size_t begin = 0;
    while(begin <= candidateId.size())
    {
        const auto end = std::min(candidateId.find('|', begin), candidateId.size());
        const auto part = candidateId.substr(begin, end - begin);
        const auto equals = part.find('=');
        if(equals == std::string::npos || equals == 0 || equals + 1 == part.size())
        {
            throw std::runtime_error("malformed composite candidate id '" + candidateId + "'");
        }
        kernels.push_back({part.substr(0, equals), part.substr(equals + 1)});
        begin = end + 1;
    }
    return kernels;
}

/// The slotted candidates of a sweep's catalog, each marked with the slots whose training rows
/// it belongs to (RFC 0021 §7).
///
/// A slot's reference pick is its first kernel in kernel-id order within the catalog. A
/// candidate is a training row of slot S when every other slot holds its reference pick, so the
/// reference program (every slot at its pick) belongs to all slots and a candidate that varies
/// two or more slots to none. The pick is not the runtime's reference (plan-build order); the
/// two agree under the separability assumption (RFC 0021 §11).
struct SlotReferences
{
    /// Slot names in slot order; empty when the catalog is not slotted.
    std::vector<std::string> slots;
    /// Per candidate, in catalog order, the slots it is a training row of.
    std::vector<std::vector<std::string>> marks;
    /// The catalog index of the reference program.
    size_t program = 0;
};

inline SlotReferences slotReferences(const std::vector<std::string>& candidateIds)
{
    std::vector<std::vector<SlotKernel>> parsed;
    parsed.reserve(candidateIds.size());
    for(const auto& id : candidateIds)
    {
        parsed.push_back(slotKernels(id));
    }
    SlotReferences references;
    if(parsed.empty() || parsed.front().empty())
    {
        if(std::any_of(
               parsed.begin(), parsed.end(), [](const auto& kernels) { return !kernels.empty(); }))
        {
            throw std::runtime_error("a catalog mixes slotted and unslotted candidate ids");
        }
        return references;
    }
    for(const auto& kernel : parsed.front())
    {
        references.slots.push_back(kernel.slot);
    }
    // Kernel ids are lowercase fixed-width UUID text, so text order is the runtime's byte order.
    std::vector<std::string> picks(references.slots.size());
    for(const auto& kernels : parsed)
    {
        if(kernels.size() != references.slots.size())
        {
            throw std::runtime_error("candidates of one catalog name different slots");
        }
        for(size_t slot = 0; slot < kernels.size(); ++slot)
        {
            if(kernels[slot].slot != references.slots[slot])
            {
                throw std::runtime_error("candidates of one catalog name different slots");
            }
            if(picks[slot].empty() || kernels[slot].kernelId < picks[slot])
            {
                picks[slot] = kernels[slot].kernelId;
            }
        }
    }
    bool found = false;
    references.marks.reserve(parsed.size());
    for(size_t index = 0; index < parsed.size(); ++index)
    {
        std::vector<std::string> varying;
        for(size_t slot = 0; slot < picks.size(); ++slot)
        {
            if(parsed[index][slot].kernelId != picks[slot])
            {
                varying.push_back(references.slots[slot]);
            }
        }
        if(varying.empty())
        {
            references.program = index;
            found = true;
            references.marks.push_back(references.slots);
        }
        else if(varying.size() == 1)
        {
            references.marks.push_back(std::move(varying));
        }
        else
        {
            references.marks.emplace_back();
        }
    }
    if(!found)
    {
        throw std::runtime_error("the catalog does not contain its reference program; "
                                 "a slotted catalog is the product of its slots' kernels");
    }
    return references;
}

/// A configuration as the knob map a candidate's variant is keyed by.
inline std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant>
    toVariantKnobs(const std::vector<hipdnn_frontend::KnobSetting>& configuration)
{
    std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant> knobs;
    for(const auto& setting : configuration)
    {
        knobs.emplace(setting.knobId(), setting.value());
    }
    return knobs;
}

/// Whether a slotted sweep's reference program (@p references.program in @p catalog) produced a
/// measurement in @p results: present, succeeded, and timed at least once. Every slot ranker
/// trains against the other slots' reference kernels, so a sweep without it is unusable.
inline bool referenceProgramTimed(const std::vector<hipdnn_frontend::AutotuneResult>& results,
                                  const hipdnn_frontend::EngineCandidatePage& catalog,
                                  const SlotReferences& references)
{
    const auto& program = catalog.candidates.at(references.program);
    const auto timed = std::find_if(
        results.begin(), results.end(), [&](const hipdnn_frontend::AutotuneResult& result) {
            return toVariantKnobs(result.knobSettings) == program.variant.knobSettings;
        });
    return timed != results.end() && timed->succeeded && timed->iterationsRun != 0;
}

/// The `slot_reference` column's text: the slot names joined by `|`, empty for a row that is
/// no slot's training row.
inline std::string slotReferenceText(const std::vector<std::string>& marks)
{
    std::string text;
    for(const auto& slot : marks)
    {
        text += (text.empty() ? "" : "|") + slot;
    }
    return text;
}

/// @p page as JSON, naming the graph by @p graphKey (@ref graphContentKey) and the id it
/// carries, @p graphId.
inline nlohmann::json pageJson(const hipdnn_frontend::EngineCandidatePage& page,
                               const std::string& graphKey,
                               const std::string& graphId)
{
    nlohmann::json candidates = nlohmann::json::array();
    for(const auto& candidate : page.candidates)
    {
        nlohmann::json entry = {{"id", candidate.id},
                                {"knob_settings", knobJson(candidate.variant.knobSettings)},
                                {"kernel_features", candidate.kernelFeatures}};
        // A slotted candidate also names each slot's kernel, in slot order; an unslotted one
        // carries no `slot_kernels`.
        if(const auto kernels = slotKernels(candidate.id); !kernels.empty())
        {
            nlohmann::json named = nlohmann::json::array();
            for(const auto& kernel : kernels)
            {
                named.push_back({{"slot", kernel.slot}, {"kernel", kernel.kernelId}});
            }
            entry["slot_kernels"] = std::move(named);
        }
        candidates.push_back(std::move(entry));
    }
    return {{"engine_id", page.engineId},
            {"graph_key", graphKey},
            {"graph_id", graphId},
            {"engine_name", page.engineName},
            {"engine_descriptor_id", page.engineDescriptorId},
            {"device_id", page.deviceId},
            {"device_arch", page.deviceArch},
            {"problem_features", page.problemFeatures},
            {"device_features", page.deviceFeatures},
            {"candidates", std::move(candidates)},
            {"total_count", page.totalCount},
            {"offset", page.offset},
            {"next_offset",
             page.nextOffset ? nlohmann::json(*page.nextOffset) : nlohmann::json(nullptr)}};
}

} // namespace hipdnn_bench
