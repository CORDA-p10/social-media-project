# Custom ranking via Module#prepend on the public timeline. Mounted at
# Mastodon's config/initializers/corda_ranker.rb; upstream tree is untouched.

module CordaRanker
  CANDIDATE_POOL_SIZE = 250
  RANKER_URL          = ENV.fetch('CORDA_RANKER_URL', 'http://127.0.0.1:8000')
  CONNECT_TIMEOUT_SEC = 3
  READ_TIMEOUT_SEC    = 60

  PAGINATION_PARAMS = %i[max_id since_id min_id].freeze

  def public_statuses
    if PAGINATION_PARAMS.any? { |k| params[k].present? }
      Rails.logger.info("[corda_ranker] paginated request — bypassing ranker")
      return super
    end

    candidates = Status.with_includes.where(id: public_feed.get(CordaRanker::CANDIDATE_POOL_SIZE).map(&:id)).to_a
    return [] if candidates.empty?

    engagements = CordaRanker.engagement_actors(candidates.map(&:id))
    payload = candidates.map { |s| CordaRanker.serialize(s, engagements[s.id]) }
    # current_account is the authenticated viewer (nil for anonymous public
    # reads — public_fetch_mode allows logged-out access). The ranker uses it
    # for per-viewer seen-rearrange; nil ⇒ a plain ranked feed.
    ordered_ids = CordaRanker.rank(
      candidates: payload,
      limit:      limit_param(self.class::DEFAULT_STATUSES_LIMIT),
      viewer_id:  current_account&.id&.to_s
    )

    by_id  = candidates.index_by { |s| s.id.to_s }
    result = ordered_ids.map { |id| by_id[id.to_s] }
    dropped = result.count(&:nil?)
    if dropped > 0
      Rails.logger.warn("[corda_ranker] dropped #{dropped} unknown ids from rank response (ranker returned ids not in candidate pool)")
    end
    result.compact
  rescue StandardError => e
    Rails.logger.warn("[corda_ranker] fallback to chronological: #{e.class}: #{e.message}")
    if defined?(candidates) && candidates.is_a?(Array)
      candidates.first(limit_param(self.class::DEFAULT_STATUSES_LIMIT))
    else
      super
    end
  end

  # Scope is nil with scope_name :current_user — AMS resolves a `current_user`
  # method on the serializer to this nil scope, so user-specific flags
  # (favourited, reblogged, bookmarked, pinned) come back blank rather than
  # raising NameError. Skipping scope_name entirely leaves `current_user`
  # undefined, which the StatusSerializer references directly. Engagement-
  # actor arrays are merged in afterwards.
  def self.serialize(status, actors = nil)
    json = ActiveModelSerializers::SerializableResource.new(
      status,
      serializer: REST::StatusSerializer,
      scope: nil,
      scope_name: :current_user,
    ).as_json
    actors ||= { favouriter_ids: [], reblogger_ids: [], replier_ids: [] }
    json.merge(actors.transform_values { |ids| ids.map(&:to_s) })
  end

  def self.engagement_actors(status_ids)
    favs = Favourite.where(status_id: status_ids)
      .pluck(:status_id, :account_id)
      .group_by(&:first)
    reblogs = Status.where(reblog_of_id: status_ids)
      .pluck(:reblog_of_id, :account_id)
      .group_by(&:first)
    replies = Status.where(in_reply_to_id: status_ids)
      .pluck(:in_reply_to_id, :account_id)
      .group_by(&:first)

    status_ids.each_with_object({}) do |sid, h|
      h[sid] = {
        favouriter_ids: (favs[sid]    || []).map(&:last).uniq,
        reblogger_ids:  (reblogs[sid] || []).map(&:last).uniq,
        replier_ids:    (replies[sid] || []).map(&:last).uniq,
      }
    end
  end

  def self.rank(candidates:, limit:, viewer_id: nil)
    response = HTTP
      .timeout(connect: CONNECT_TIMEOUT_SEC, read: READ_TIMEOUT_SEC)
      .post(
        "#{RANKER_URL}/rank",
        json: { candidates: candidates, limit: limit, viewer_id: viewer_id }
      )
    raise "ranker responded #{response.code}" unless response.status.success?
    ids = JSON.parse(response.body.to_s)['ordered_ids']
    raise 'ranker returned no ordered_ids' unless ids.is_a?(Array)
    ids
  end
end

Rails.configuration.to_prepare do
  Api::V1::Timelines::PublicController.prepend(CordaRanker)
  Rails.logger.info("[corda_ranker] prepended on PublicController (ranker_url=#{CordaRanker::RANKER_URL}, connect=#{CordaRanker::CONNECT_TIMEOUT_SEC}s, read=#{CordaRanker::READ_TIMEOUT_SEC}s, pool=#{CordaRanker::CANDIDATE_POOL_SIZE})")
end
