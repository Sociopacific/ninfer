#include "core/cache_archive.h"
#include <chrono>
#include <string>
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {


SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept = default;

SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;

SequencePlan::~SequencePlan() = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept = default;

SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;

SequencePlanner::~SequencePlanner() = default;

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan(detail::finalize_sequence_plan_impl(std::move(impl_), main_page_groups));
}

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept = default;

RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;

RequestBasePlan::~RequestBasePlan() = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const PreparedContextCache empty;
    return impl_ != nullptr ? impl_->context_cache : empty;
}

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prefix_digests.size()) {
        return std::nullopt;
    }
    return PrefixShortlistKey{
        .digests      = impl_->prefix_digests.at(frontier),
        .frontier     = frontier,
        .identity_tag = impl_->prefix_identity_tag,
    };
}

std::optional<runtime::PrefillWork>
RequestBasePlan::shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr) { return std::nullopt; }
    const auto found = std::find_if(impl_->shared_candidates.begin(),
                                    impl_->shared_candidates.end(), [&](const auto& candidate) {
                                        return candidate.frontier == frontier && candidate.identity;
                                    });
    return found == impl_->shared_candidates.end()
               ? std::nullopt
               : std::optional<runtime::PrefillWork>(found->identity->rebuild_work);
}

PressurePlanningSession::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept
    : impl_(std::move(impl)) {}

PressurePlanningSession::PressurePlanningSession(PressurePlanningSession&&) noexcept = default;

PressurePlanningSession&
PressurePlanningSession::operator=(PressurePlanningSession&&) noexcept = default;

PressurePlanningSession::~PressurePlanningSession() = default;

CapturePressurePlanningSession::CapturePressurePlanningSession(
    CapturePressurePlanningSession&&) noexcept = default;

CapturePressurePlanningSession&
CapturePressurePlanningSession::operator=(CapturePressurePlanningSession&&) noexcept = default;

CapturePressurePlanningSession::~CapturePressurePlanningSession() = default;

PressureTargetHandle
PressurePlanningSession::identity_target(runtime::PlanningCandidateId candidate) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->identity_target(candidate);
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(runtime::PlanningCandidateId root_candidate) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->root_maximal_target(root_candidate);
}

PressureTargetHandle
PressurePlanningSession::maximal_target(runtime::PlanningCandidateId candidate) {
    return impl_->maximal_target(candidate);
}

PressureConstructionCursor PressurePlanningSession::begin_construction(PressureTargetHandle target,
                                                                       bool restore) {
    return impl_->begin_construction(target, restore);
}

runtime::PressureConstructionStep
PressurePlanningSession::next_construction_option(PressureConstructionCursor& cursor) {
    return impl_->next_construction_option(cursor);
}

void PressurePlanningSession::choose_construction(PressureConstructionCursor& cursor,
                                                  runtime::PressureConstructionOptionId option) {
    impl_->choose_construction(cursor, option);
}

std::optional<PressureTargetHandle>
PressurePlanningSession::construction_target(const PressureConstructionCursor& cursor) {
    return impl_->construction_target(cursor);
}

runtime::PressureTargetGuidance PressurePlanningSession::guidance(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->guidance(target);
}

AssessedPressureTarget PressurePlanningSession::assess(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->assess(target);
}

PreparedPressureExpansion PressurePlanningSession::prepare_expansion(PressureTargetHandle parent,
                                                                     std::uint32_t maximum_owners) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->prepare_expansion(parent, maximum_owners);
}

PressureExpansionView
PressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->commit_expansion(std::move(prepared));
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&& prepared) noexcept {
    if (impl_ != nullptr) { impl_->discard_expansion(std::move(prepared)); }
}

runtime::PrefillWork PressurePlanningSession::shared_capture_split_prefill_work(
    const AssessedPressureTarget& assessed, const PreparedPrompt& prompt,
    std::span<const std::uint32_t> frontiers) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->shared_capture_split_prefill_work(assessed, PreparedPromptAccess::view(prompt),
                                                    frontiers);
}

std::optional<ResourcePlan> PressurePlanningSession::seal(AssessedPressureTarget&& assessed,
                                                          const PreparedPrompt& prompt,
                                                          runtime::FinalScheduleIntent intent) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<AdmissionCandidate> sealed =
        impl_->seal(std::move(assessed), PreparedPromptAccess::view(prompt), intent);
    if (!sealed) { return std::nullopt; }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan(std::move(*sealed), impl_->resource_revision, needs_transfer);
}

std::optional<CapturePressurePlan>
PressurePlanningSession::seal_capture(AssessedPressureTarget&& assessed) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<CapturePressureCandidate> sealed = impl_->seal_capture(std::move(assessed));
    if (!sealed) { return std::nullopt; }
    return CapturePressurePlan(std::move(*sealed), impl_->resource_revision);
}

PressureTargetHandle CapturePressurePlanningSession::identity_target() const {
    if (!candidate_.impl_) { throw std::logic_error("capture pressure candidate is empty"); }
    return session_.identity_target(candidate_id());
}

runtime::PressureTargetGuidance
CapturePressurePlanningSession::guidance(PressureTargetHandle target) {
    return session_.guidance(target);
}

AssessedPressureTarget CapturePressurePlanningSession::assess(PressureTargetHandle target) {
    return session_.assess(target);
}

PreparedPressureExpansion
CapturePressurePlanningSession::prepare_expansion(PressureTargetHandle parent) {
    return session_.prepare_expansion(parent);
}

PressureExpansionView
CapturePressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    return session_.commit_expansion(std::move(prepared));
}

void CapturePressurePlanningSession::discard_expansion(
    PreparedPressureExpansion&& prepared) noexcept {
    session_.discard_expansion(std::move(prepared));
}

std::optional<CapturePressurePlan>
CapturePressurePlanningSession::seal(AssessedPressureTarget&& assessed) {
    return session_.seal_capture(std::move(assessed));
}

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}

Program::~Program() noexcept = default;

RequestBasePlan Program::plan_request(const PreparedPrompt& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(PreparedPromptAccess::view(prompt), options);
}

std::vector<float> Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target) {
    return impl_->causal_score(PreparedPromptAccess::take(std::move(prompt)), first_target);
}

std::optional<AdmissionCandidate> Program::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId destination,
    const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source) {
    return impl_->inspect_admission(PreparedPromptAccess::view(prompt), base, destination, source,
                                    shared_source, checkpoint, must_retain_private_source);
}

std::optional<ResourcePlan> Program::seal_identity(const AdmissionCandidate& admission,
                                                   const PreparedPrompt& prompt,
                                                   runtime::FinalScheduleIntent intent) {
    std::optional<AdmissionCandidate> sealed = impl_->seal_materialization(
        admission, PreparedPromptAccess::view(prompt), {}, {}, {}, {}, {}, {});
    if (!sealed) { return std::nullopt; }
    impl_->select_shared_captures(*sealed, PreparedPromptAccess::view(prompt),
                                  intent.shared_capture_frontiers);
    if (impl_->revalidate_materialization(*sealed, PreparedPromptAccess::view(prompt)) !=
        runtime::PreflightStatus::Ready) {
        return std::nullopt;
    }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan(std::move(*sealed), impl_->resource_revision(), needs_transfer);
}

PressurePlanningSession
Program::begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                                 std::span<const runtime::PlanningCandidateId> candidate_ids,
                                 std::span<const ContinuationHandle* const> private_owners,
                                 std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                 std::span<const SharedPrefixHandle* const> shared_owners,
                                 std::span<const runtime::PlanningOwnerId> shared_owner_ids) {
    using SessionImpl = detail::PressurePlanningSessionImpl;
    std::vector<SessionImpl::PhysicalCandidateBinding> physical_candidates;
    physical_candidates.reserve(candidates.size());
    for (const AdmissionCandidate* candidate : candidates) {
        if (candidate == nullptr || candidate->impl_ == nullptr) {
            throw std::invalid_argument("pressure planning candidate is empty");
        }
        physical_candidates.push_back(SessionImpl::PhysicalCandidateBinding{
            .state     = candidate->impl_.get(),
            .admission = candidate->impl_.get(),
        });
    }
    return PressurePlanningSession(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, physical_candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids));
}

runtime::PrefillWork
Program::shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                           const PreparedPrompt& prompt,
                                           std::span<const std::uint32_t> frontiers) {
    if (impl_ == nullptr) { throw std::logic_error("Program is empty"); }
    return impl_->shared_capture_split_prefill_work(candidate, PreparedPromptAccess::view(prompt),
                                                    frontiers);
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
    if (plan.revision_.value == 0 || plan.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->reserve_materialization(
        std::move(plan.admission_), PreparedPromptAccess::take(std::move(prompt)), cancellation);
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& blocked_head,
                                   const ResourcePlan& candidate,
                                   std::span<const SequenceHandle> persistent_borrowers) const {
    if (candidate.revision_.value == 0 || candidate.revision_ != impl_->resource_revision() ||
        !impl_->persistent_backfill_safe(blocked_head, candidate.admission_,
                                         persistent_borrowers)) {
        return std::nullopt;
    }
    return PersistentBackfillProof(candidate.revision_);
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    return impl_->progress_context_transaction(cancellation);
}

void Program::finalize_context_transaction() noexcept { impl_->finalize_context_transaction(); }

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle sequence,
                                         runtime::ExecutionTiming* failed_timing) {
    return impl_->advance_prefill(sequence, failed_timing);
}

CaptureAssessment
Program::inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                         const SharedPrefixHandle* replacement,
                         std::optional<runtime::CheckpointRef> private_replacement,
                         bool permit_shared_publication) const {
    return impl_->inspect_capture(offer, exact_shared, replacement, private_replacement,
                                  permit_shared_publication);
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const ContinuationHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

CapturePressurePlanningSession Program::begin_capture_pressure_planning(
    const CaptureAssessment& assessment, std::span<const ContinuationHandle* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids) {
    CapturePressureCandidate candidate(impl_->make_capture_physical_candidate(assessment));
    using SessionImpl = detail::PressurePlanningSessionImpl;
    const std::array physical_candidates{SessionImpl::PhysicalCandidateBinding{
        .state   = candidate.impl_.get(),
        .capture = candidate.impl_.get(),
    }};
    const std::array candidate_ids{CapturePressurePlanningSession::candidate_id()};
    PressurePlanningSession session(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, physical_candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids));
    return CapturePressurePlanningSession(std::move(candidate), std::move(session));
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const SharedPrefixHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

bool Program::shared_capture_matches(const CaptureOffer& offer,
                                     const SharedPrefixHandle& shared) const {
    return impl_->shared_capture_matches(offer, shared);
}

void Program::skip_capture(CaptureOffer&& offer) { impl_->skip_capture(std::move(offer)); }

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                bool permit_shared_publication,
                                runtime::CancellationFlagView cancellation) {
    return impl_->reserve_active_capture(std::move(offer), exact_shared, replacement,
                                         private_replacement, permit_shared_publication,
                                         cancellation);
}

runtime::ContextTransactionReserveStatus Program::reserve_active_capture_with_pressure(
    CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
    const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    CapturePressurePlan&& pressure, runtime::CancellationFlagView cancellation) {
    if (pressure.revision_.value == 0 || pressure.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->reserve_active_capture_with_pressure(
        std::move(offer), exact_shared, replacement, private_replacement, permit_shared_publication,
        std::move(pressure.pressure_), cancellation);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing) {
    return impl_->decode(sequences, budgets, failed_timing);
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                              runtime::ExecutionTiming* failed_timing) {
    return impl_->append_forced_tokens(sequences, row_major_tokens, row_stride,
                                       prefix_execution_splits, failed_timing);
}

CommitResult Program::commit(PendingBatch&& pending,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation observation,
                             runtime::ExecutionTiming* failed_timing) {
    return impl_->commit(std::move(pending), decisions, observation, failed_timing);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    return impl_->abort_pending(std::move(pending));
}

FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }

AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }

ReleaseResult Program::release_continuation(ContinuationHandle&& continuation) noexcept {
    return impl_->release_continuation(std::move(continuation));
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& shared) noexcept {
    return impl_->release_shared_prefix(std::move(shared));
}

void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    return impl_->isolated_request_feasible(base);
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return impl_->resource_revision();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return SequencePlanner(detail::make_sequence_planner_impl(parameters, device, options));
}

std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        SequencePlan&& plan, DeviceContext& device,
                                        const StartupObserver& startup_observer) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->parameters != &parameters) {
        throw std::invalid_argument("sequence plan belongs to another model instance");
    }
    if (plan.impl_->multiprocessor_count != device.multiprocessor_count()) {
        throw std::invalid_argument("sequence plan device capacity does not match execution");
    }
    auto impl =
        std::make_unique<detail::ProgramImpl>(parameters, *plan.impl_, device, startup_observer);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5 {

CaptureAssessment::CaptureAssessment()
    : implementation(std::make_shared<detail::CaptureAssessmentImpl>()) {}

AdmissionCandidate::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

AdmissionCandidate::AdmissionCandidate(AdmissionCandidate&&) noexcept = default;

AdmissionCandidate& AdmissionCandidate::operator=(AdmissionCandidate&&) noexcept = default;

AdmissionCandidate::~AdmissionCandidate() = default;

CapturePressureCandidate::CapturePressureCandidate(
    std::unique_ptr<detail::CapturePressureCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

CapturePressureCandidate::CapturePressureCandidate(CapturePressureCandidate&&) noexcept = default;

CapturePressureCandidate&
CapturePressureCandidate::operator=(CapturePressureCandidate&&) noexcept = default;

CapturePressureCandidate::~CapturePressureCandidate() = default;

const runtime::RequestPlanSummary& AdmissionCandidate::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const runtime::IdentityMaterializationAssessment&
AdmissionCandidate::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty;
    return impl_ != nullptr ? impl_->identity_assessment : empty;
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
namespace {
using cache_archive::Reader;
using cache_archive::Writer;
const std::string cache_run_id = std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
std::string page_group_name(LogicalKVPageStore& store, std::span<const LogicalKVPageHandle> pages, bool backend) {
    std::uint64_t key = 1469598103934665603ULL;
    for (auto page : pages) {
        key = (key ^ store.descriptor_index(page)) * 1099511628211ULL;
        key = (key ^ store.content_epoch(page)) * 1099511628211ULL;
        key = (key ^ store.committed_columns(page)) * 1099511628211ULL;
    }
    return std::string(backend ? "mtp-" : "kv-") + cache_run_id + "-" + std::to_string(key) + ".bin";
}
void write_payload(const std::filesystem::path& path, const std::byte* bytes, std::size_t count) {
    const cache_archive::PayloadSpan span{bytes,count};
    if (!cache_archive::stage_payload(path.string()+".tmp", std::span(&span,1))) {
        Writer file(path.string() + ".tmp"); file(std::uint64_t(count)); file.bytes(bytes, count); file.close();
    }
    if (cache_archive::capture_stats) cache_archive::capture_stats->payload_bytes_written += count;
    cache_archive::publish(path.string() + ".tmp", path);
}
void read_payload(const std::filesystem::path& path, std::byte* bytes, std::size_t count) {
    Reader file(path); std::uint64_t size; file(size);
    if (size != count) throw std::runtime_error("cache payload layout mismatch");
    file.bytes(bytes, count);
}
}

void ProgramImpl::save_continuation(const ContinuationHandle& continuation, const std::filesystem::path& path) {
    cache_archive::PayloadCaptureBarrier capture_barrier;
    if (has_context_transaction() || pending_transaction_ || !valid_continuation(continuation))
        throw std::logic_error("cache checkpoint requires a settled continuation");
    SequenceState& state = continuation_states[ContractAccess::index(continuation)];
    if (!state.kv || !host_kv_extents || !host_state_images)
        throw std::runtime_error("cache checkpoint requires Host KV and state pools");
    device.synchronize();
    Writer ar(path.string() + ".tmp");
    ar(std::uint64_t(0x4e494e4341434831ULL));
    ar(state.execution_frontier); ar(state.ledger_frontier); ar(state.ledger);
    state.prefix_identity.archive(ar); state.prefix_digests.archive(ar);
    ar(state.rope_delta); ar(state.text_kv_valid); ar(state.mtp_kv_valid);
    ar(state.dflash_context_frontier); ar(state.mtp_drafts); ar(state.mtp_draft_count);
    ar(state.tail_hidden_valid); ar(state.endpoint_valid); ar(state.rewrite_checkpoint);
    ar(state.rebuild_work); ar(state.rebuild_tail_begin);
    std::vector<StateImageHandle> images;
    auto image_id = [&](StateImageHandle handle) -> std::int32_t {
        if (!state_store->valid(handle)) return -1;
        auto found = std::find(images.begin(), images.end(), handle);
        if (found != images.end()) return static_cast<std::int32_t>(found - images.begin());
        images.push_back(handle); return static_cast<std::int32_t>(images.size()-1);
    };
    ar(image_id(state.state.read)); ar(image_id(state.state.write));
    ar(state.rewrite_state ? image_id(*state.rewrite_state) : -1);
    ar(std::uint64_t(state.long_anchors.size()));
    for (auto& anchor : state.long_anchors) {
        ar(image_id(anchor.state)); ar(anchor.frontier); ar(anchor.ordinal); ar(anchor.rebuild_work);
    }
    ar(std::uint64_t(images.size()));
    for (auto image : images) {
        std::string name = "state-" + cache_run_id + "-" + std::to_string(state_store->descriptor_index(image)) +
                           "-" + std::to_string(state_store->content_epoch(image)) + ".bin";
        const std::string key=(path.parent_path()/name).string();
        if (auto saved=checkpoint_payload_names.find(key); saved!=checkpoint_payload_names.end()) name=saved->second;
        ar(name);
        if (cache_archive::payload_exists(path.parent_path()/name)) continue;
        if (state_store->residency(image) == StateReplicaResidency::DeviceOnly) {
            CudaEventTimer timer(device,device.transfer_stream); timer.start();
            auto transfer = state_store->begin_device_to_host(image, device.transfer_stream);
            if (!transfer) throw std::runtime_error("checkpoint Host state pool exhausted");
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            const double seconds=timer.stop_ms()/1000.0;
            state_store->publish_transfer(std::move(*transfer), true);
            if (cache_archive::capture_stats) {
                cache_archive::capture_stats->device_to_host_bytes += host_state_images->layout().image_bytes;
                cache_archive::capture_stats->device_to_host_seconds += seconds;
            }
        }
        auto view = state_store->checkpoint_host_view(image);
        write_payload(path.parent_path()/name, view.data, view.layout->image_bytes);
    }
    auto save_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages, KVAddressSpaceHandle address, bool backend) {
        const std::uint32_t frontier = addresses.committed_frontier(address);
        const std::uint32_t count = frontier ? (frontier + kPagedKVPageSize - 1) / kPagedKVPageSize : 0;
        const std::size_t stride = host_kv_extents->page_layout(pages).page_stride;
        ar(frontier); ar(std::uint64_t(stride)); ar(count);
        for (std::uint32_t begin = 0; begin < count; begin += 64) {
            std::vector<LogicalKVPageHandle> group;
            for (std::uint32_t index = begin; index < std::min(count, begin+64); ++index)
                group.push_back(addresses.logical_page(address, index));
            std::string name = page_group_name(pages, group, backend);
            const std::string key=(path.parent_path()/name).string();
            if (auto saved=checkpoint_payload_names.find(key); saved!=checkpoint_payload_names.end()) name=saved->second;
            ar(name);
            for (auto page : group) ar(pages.committed_columns(page));
            if (cache_archive::payload_exists(path.parent_path()/name)) continue;
            std::vector<LogicalKVPageHandle> missing;
            for (auto page : group) if (!pages.host_resident(page)) missing.push_back(page);
            if (!missing.empty()) {
                auto reservation = host_kv_extents->prepare(pages, missing);
                if (!reservation) throw std::runtime_error("checkpoint Host KV pool exhausted");
                auto sources = host_kv_extents->device_sources(*reservation);
                CudaEventTimer timer(device,device.transfer_stream); timer.start();
                pages.physical_pool().copy_to_host(sources, host_kv_extents->writable_view(*reservation), device.transfer_stream);
                CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
                const double seconds=timer.stop_ms()/1000.0;
                (void)host_kv_extents->publish(std::move(*reservation));
                if (cache_archive::capture_stats) {
                    cache_archive::capture_stats->device_to_host_bytes += stride*missing.size();
                    cache_archive::capture_stats->device_to_host_seconds += seconds;
                }
            }
            std::vector<cache_archive::PayloadSpan> spans;
            spans.reserve(group.size());
            for (auto page : group) {
                auto replica = pages.host_replica(page);
                auto view = host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
                spans.push_back({view.data(),stride});
            }
            const auto temporary=(path.parent_path()/name).string()+".tmp";
            if (!cache_archive::stage_payload(temporary,spans)) {
                Writer payload(temporary); payload(std::uint64_t(stride * group.size()));
                for (auto span : spans) payload.bytes(span.data,span.size);
                payload.close();
            }
            if (cache_archive::capture_stats) cache_archive::capture_stats->payload_bytes_written += stride*group.size();
            cache_archive::publish(temporary, path.parent_path()/name);
        }
    };
    save_kv(*text_kv_addresses, *text_kv_pages, state.kv->text, false);
    ar(state.kv->backend.has_value());
    if (state.kv->backend) save_kv(*backend_kv_addresses, *backend_kv_pages, *state.kv->backend, true);
    cache_archive::finish_payload_capture();
    ar.close(); cache_archive::publish(path.string()+".tmp", path);
    advance_resource_revision();
}

FinishResult ProgramImpl::load_continuation(const std::filesystem::path& path) {
    if (has_context_transaction() || pending_transaction_ || !host_state_images || !host_kv_extents)
        throw std::logic_error("cache restore requires settled Host storage");
    auto slot = allocate_continuation_slot();
    if (!slot) throw std::runtime_error("cache continuation slots exhausted");
    SequenceState& state = continuation_states[*slot];
    std::vector<StateImageHandle> images;
    try {
        Reader ar(path); std::uint64_t magic; ar(magic);
        if (magic != 0x4e494e4341434831ULL) throw std::runtime_error("invalid cache checkpoint format");
        ar(state.execution_frontier); ar(state.ledger_frontier); ar(state.ledger);
        state.prefix_identity.archive(ar); state.prefix_digests.archive(ar);
        ar(state.rope_delta); ar(state.text_kv_valid); ar(state.mtp_kv_valid);
        ar(state.dflash_context_frontier); ar(state.mtp_drafts); ar(state.mtp_draft_count);
        ar(state.tail_hidden_valid); ar(state.endpoint_valid); ar(state.rewrite_checkpoint);
        ar(state.rebuild_work); ar(state.rebuild_tail_begin);
        if (state.execution_frontier > capacity || state.ledger_frontier != state.ledger.size())
            throw std::runtime_error("checkpoint token frontier exceeds capacity");
        std::int32_t read_id, write_id, rewrite_id; ar(read_id); ar(write_id); ar(rewrite_id);
        std::uint64_t anchors; ar(anchors);
        if (anchors > 64) throw std::runtime_error("invalid checkpoint anchor count");
        std::vector<std::int32_t> anchor_ids(anchors); state.long_anchors.resize(anchors);
        for (std::size_t i=0; i<anchors; ++i) {
            ar(anchor_ids[i]); ar(state.long_anchors[i].frontier); ar(state.long_anchors[i].ordinal); ar(state.long_anchors[i].rebuild_work);
        }
        std::uint64_t image_count; ar(image_count);
        if (image_count > anchors+3) throw std::runtime_error("invalid checkpoint state count");
        std::vector<std::byte> image(host_state_images->layout().image_bytes);
        for (std::size_t i=0; i<image_count; ++i) {
            std::string name; ar(name); read_payload(path.parent_path()/name, image.data(), image.size());
            auto handle=state_store->import_checkpoint(image.data(), image.size());
            images.push_back(handle);
            std::string key="state-" + cache_run_id + "-" + std::to_string(state_store->descriptor_index(handle)) +
                            "-" + std::to_string(state_store->content_epoch(handle)) + ".bin";
            checkpoint_payload_names[(path.parent_path()/key).string()]=name;
        }
        auto get_image = [&](std::int32_t id) {
            if (id < 0 || static_cast<std::size_t>(id) >= images.size()) throw std::runtime_error("invalid checkpoint state reference");
            return images[id];
        };
        if (state.endpoint_valid) state.state = ActiveStateBinding{.read=get_image(read_id), .write=get_image(write_id)};
        if (rewrite_id >= 0) { state.rewrite_state = get_image(rewrite_id); state_store->retain_checkpoint_reference(*state.rewrite_state); }
        for (std::size_t i=0; i<anchors; ++i) {
            state.long_anchors[i].state = get_image(anchor_ids[i]);
            state_store->retain_checkpoint_reference(state.long_anchors[i].state);
        }
        auto load_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages, bool backend) {
            std::uint32_t frontier, count; std::uint64_t stride; ar(frontier); ar(stride); ar(count);
            if (frontier > capacity || count != (frontier+kPagedKVPageSize-1)/kPagedKVPageSize ||
                stride != host_kv_extents->page_layout(pages).page_stride)
                throw std::runtime_error("checkpoint KV layout mismatch");
            std::vector<LogicalKVPageHandle> imported;
            std::vector<HostKVExtentCapability> extents;
            imported.reserve(count); extents.reserve((count+63)/64);
            struct ImportJob {
                std::string name;
                std::vector<LogicalKVPageHandle> group;
                std::unique_ptr<HostKVExtentReservation> reservation;
                std::future<void> read;
            };
            std::deque<ImportJob> jobs;
            auto finish_job = [&] {
                auto job=std::move(jobs.front()); jobs.pop_front();
                job.read.get();
                extents.push_back(host_kv_extents->publish(std::move(*job.reservation)));
                checkpoint_payload_names[(path.parent_path()/page_group_name(pages,job.group,backend)).string()]=job.name;
            };
            try {
                for (std::uint32_t begin=0; begin<count; begin+=64) {
                    std::string name; ar(name);
                    std::vector<LogicalKVPageHandle> group;
                    group.reserve(std::min(count-begin,64U));
                    for (std::uint32_t index=begin; index<std::min(count,begin+64); ++index) {
                        std::uint32_t columns; ar(columns);
                        if (columns < std::min(std::uint32_t(kPagedKVPageSize),frontier-index*std::uint32_t(kPagedKVPageSize)))
                            throw std::runtime_error("checkpoint KV coverage is incomplete");
                        auto handle=pages.import_host_page(columns);
                        imported.push_back(handle); group.push_back(handle);
                    }
                    auto reservation = host_kv_extents->prepare(pages, group, true);
                    if (!reservation) throw std::runtime_error("checkpoint Host KV capacity exhausted");
                    auto view = host_kv_extents->writable_view(*reservation);
                    const auto filename=path.parent_path()/name;
                    const auto bytes=stride*group.size();
                    jobs.push_back(ImportJob{std::move(name),std::move(group),
                        std::make_unique<HostKVExtentReservation>(std::move(*reservation)),
                        std::async(std::launch::async,[filename,data=view.data(),bytes] {
                            read_payload(filename,data,bytes);
                        })});
                    if (jobs.size() >= 8) finish_job();
                }
                while (!jobs.empty()) finish_job();
                return addresses.import_host_pages(imported, frontier);
            } catch (...) {
                // Async readers finish before reservations and logical pages can be reclaimed.
                jobs.clear();
                // Host-only replicas cannot be detached while their logical references exist.
                // Drop those references first; extents then reclaim the zero-reference descriptors.
                for (auto page : imported) (void)pages.release(page);
                for (auto extent : extents) (void)host_kv_extents->release(extent);
                throw;
            }
        };
        state.kv.emplace(SequenceKVBundle{.text=load_kv(*text_kv_addresses,*text_kv_pages,false)});
        bool backend; ar(backend);
        if (backend) {
            if (!backend_kv_addresses || !backend_kv_pages) throw std::runtime_error("checkpoint speculative mode mismatch");
            state.kv->backend=load_kv(*backend_kv_addresses,*backend_kv_pages,true);
        }
        continuation_slots[*slot].role = ContinuationSlotRole::Catalogued;
        FinishResult result; populate_continuation_summary(state, result.summary);
        result.summary.active_references = 0;
        result.continuation.emplace(ContractAccess::make_continuation(this,*slot,continuation_slots[*slot].generation));
        result.status = runtime::ConsumeStatus::Consumed; result.disposition=runtime::FinishDisposition::Catalogued;
        advance_resource_revision(); return result;
    } catch (...) {
        release_continuation_slot_best_effort(*slot);
        for (auto image : images) if (state_store->valid(image) && !state_store->checkpoint_references(image)) (void)state_store->release(image);
        throw;
    }
}
}
namespace ninfer::models::qwen3_5 {
void Program::save_continuation(const ContinuationHandle& continuation, const std::filesystem::path& path) { impl_->save_continuation(continuation,path); }
FinishResult Program::load_continuation(const std::filesystem::path& path) { return impl_->load_continuation(path); }
}
