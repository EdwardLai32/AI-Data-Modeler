"use client";

import { Badge, Chip } from "@/components/ui/Badge";
import { Panel } from "@/components/ui/Panel";
import { DataBar, StatTile } from "@/components/ui/Metrics";
import { Disclosure } from "@/components/ui/Rationale";
import { EmptyState } from "@/components/ui/States";
import { bytes, compactNumber, EMPTY, integer, metricValue, percent } from "@/lib/format";
import { humanise } from "@/lib/format";
import { severityTone } from "@/lib/labels";
import type { CorrelationPair, DatasetProfile } from "@/types/api";

/**
 * Deterministic dataset facts.
 *
 * Everything in this panel was computed by pandas in `profiling/profiler.py`;
 * no agent touched any of it. That is why it is shown before the narrative
 * panels — it is the evidence the agents were reasoning over.
 */

function CorrelationRow({ pair, target }: { pair: CorrelationPair; target: string | null }) {
  const other = target && pair.left === target ? pair.right : pair.left === target ? pair.right : pair.left;
  const label = target ? other : `${pair.left} ↔ ${pair.right}`;
  const magnitude = Math.min(1, Math.abs(pair.coefficient));
  return (
    <DataBar
      name={<span className="truncate font-mono text-[11px]">{label}</span>}
      nameTitle={`${pair.left} ↔ ${pair.right} (${pair.method})`}
      fraction={magnitude}
      // Sign is polarity, so the two directions take the diverging pair's poles;
      // the signed number is always written out beside the bar.
      tone={pair.coefficient < 0 ? "critical" : "accent"}
      valueText={`${pair.coefficient > 0 ? "+" : ""}${pair.coefficient.toFixed(3)}`}
    />
  );
}

export function ProfilePanel({ profile }: { profile: DatasetProfile | null }) {
  if (!profile) {
    return (
      <Panel title="Dataset profile" subtitle="Computed by pandas, not by an agent">
        <EmptyState
          title="Not profiled yet"
          hint="The profiler runs immediately after ingestion and fills this panel in."
        />
      </Panel>
    );
  }

  const kindCounts = new Map<string, number>();
  for (const column of profile.columns) {
    kindCounts.set(column.kind, (kindCounts.get(column.kind) ?? 0) + 1);
  }
  const target = profile.target;
  const maxClassCount = target?.class_counts.reduce((max, item) => Math.max(max, item.count), 0) ?? 0;

  return (
    <Panel
      title="Dataset profile"
      subtitle={`Computed in ${profile.profile_seconds.toFixed(2)}s · dataset ${profile.dataset_id}`}
      bodyClassName="space-y-4"
    >
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 xl:grid-cols-5">
        <StatTile label="Rows" value={compactNumber(profile.n_rows)} note={integer(profile.n_rows)} />
        <StatTile label="Columns" value={integer(profile.n_columns)} />
        <StatTile
          label="Missing cells"
          value={percent(profile.missing_cell_fraction)}
          note={`${integer(profile.total_missing_cells)} cells`}
        />
        <StatTile
          label="Duplicate rows"
          value={percent(profile.duplicate_fraction)}
          note={`${integer(profile.n_duplicate_rows)} rows`}
        />
        <StatTile label="In memory" value={bytes(profile.memory_bytes)} />
      </div>

      <div className="flex flex-wrap gap-1.5">
        {[...kindCounts.entries()]
          .sort((a, b) => b[1] - a[1])
          .map(([kind, count]) => (
            <Chip key={kind}>
              {humanise(kind)} · {count}
            </Chip>
          ))}
      </div>

      {target ? (
        <div className="rounded-lg border border-hairline bg-surface-2 px-3.5 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-xs font-semibold text-ink">Target</h3>
            <Chip mono>{target.name}</Chip>
            <Chip>{humanise(target.kind)}</Chip>
            {target.n_classes !== null ? <Chip>{target.n_classes} classes</Chip> : null}
            {target.is_imbalanced ? (
              <Badge tone="warning">
                Imbalanced {target.imbalance_ratio !== null ? `${target.imbalance_ratio.toFixed(1)}:1` : ""}
              </Badge>
            ) : null}
            {target.n_missing > 0 ? (
              <Badge tone="serious">{integer(target.n_missing)} missing</Badge>
            ) : null}
          </div>

          {target.class_counts.length ? (
            <div className="mt-2">
              {target.class_counts.slice(0, 8).map((item) => (
                <DataBar
                  key={item.value}
                  name={<span className="truncate font-mono text-[11px]">{item.value}</span>}
                  nameTitle={item.value}
                  fraction={maxClassCount ? item.count / maxClassCount : 0}
                  valueText={`${integer(item.count)} · ${percent(item.fraction)}`}
                />
              ))}
            </div>
          ) : (
            <dl className="mt-2 grid grid-cols-3 gap-2 text-xs">
              <div>
                <dt className="text-ink-3">Mean</dt>
                <dd className="tabular text-ink">{metricValue(target.mean)}</dd>
              </div>
              <div>
                <dt className="text-ink-3">Std</dt>
                <dd className="tabular text-ink">{metricValue(target.std)}</dd>
              </div>
              <div>
                <dt className="text-ink-3">Skewness</dt>
                <dd className="tabular text-ink">{metricValue(target.skewness)}</dd>
              </div>
            </dl>
          )}
        </div>
      ) : null}

      {profile.leakage_findings.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Leakage suspects</h3>
          <ul className="mt-1.5 space-y-1.5">
            {profile.leakage_findings.map((finding) => (
              <li
                key={finding.column}
                className="rounded-md border border-hairline bg-surface-2 px-3 py-2"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <Chip mono>{finding.column}</Chip>
                  <Badge tone={severityTone(finding.severity)}>{humanise(finding.severity)}</Badge>
                  <span className="tabular text-[11px] text-ink-3">
                    {finding.method} {finding.score.toFixed(3)}
                  </span>
                </div>
                <p className="prose-agent mt-1 text-xs leading-relaxed text-ink-2">
                  {finding.reason}
                </p>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {profile.quality_issues.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Quality issues</h3>
          <ul className="mt-1.5 space-y-1">
            {profile.quality_issues.map((issue, index) => (
              <li key={`${issue.code}-${index}`} className="flex flex-wrap items-baseline gap-2">
                <Badge tone={severityTone(issue.severity)}>{humanise(issue.severity)}</Badge>
                <span className="font-mono text-[11px] text-ink-3">{issue.code}</span>
                <span className="min-w-0 flex-1 text-xs leading-relaxed text-ink-2">
                  {issue.detail}
                </span>
                {issue.columns.length ? (
                  <span className="text-[11px] text-ink-3">
                    {issue.columns.slice(0, 4).join(", ")}
                    {issue.columns.length > 4 ? ` +${issue.columns.length - 4}` : ""}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {profile.target_correlations.length ? (
        <div>
          <h3 className="text-xs font-semibold text-ink">Correlation with the target</h3>
          <div className="mt-1">
            {profile.target_correlations.slice(0, 10).map((pair, index) => (
              <CorrelationRow key={`${pair.left}-${pair.right}-${index}`} pair={pair} target={target?.name ?? null} />
            ))}
          </div>
        </div>
      ) : null}

      {profile.highly_correlated_pairs.length ? (
        <Disclosure summary="Highly correlated feature pairs" count={profile.highly_correlated_pairs.length}>
          <div>
            {profile.highly_correlated_pairs.slice(0, 15).map((pair, index) => (
              <CorrelationRow key={`${pair.left}-${pair.right}-${index}`} pair={pair} target={null} />
            ))}
          </div>
        </Disclosure>
      ) : null}

      <Disclosure summary="All columns" count={profile.columns.length}>
        <div className="max-h-80 overflow-auto rounded-lg border border-hairline">
          <table className="w-full text-left text-[11px]">
            <thead className="sticky top-0 bg-surface-2 text-ink-3">
              <tr>
                <th className="px-2.5 py-1.5 font-medium">Column</th>
                <th className="px-2.5 py-1.5 font-medium">Kind</th>
                <th className="px-2.5 py-1.5 font-medium">Dtype</th>
                <th className="px-2.5 py-1.5 text-right font-medium">Missing</th>
                <th className="px-2.5 py-1.5 text-right font-medium">Unique</th>
                <th className="px-2.5 py-1.5 font-medium">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hairline">
              {profile.columns.map((column) => (
                <tr key={column.name} className="text-ink-2">
                  <td className="max-w-[12rem] truncate px-2.5 py-1 font-mono text-ink" title={column.name}>
                    {column.name}
                  </td>
                  <td className="px-2.5 py-1">{humanise(column.kind)}</td>
                  <td className="px-2.5 py-1 font-mono">{column.dtype}</td>
                  <td className="tabular px-2.5 py-1 text-right">
                    {column.n_missing ? percent(column.missing_fraction) : "0%"}
                  </td>
                  <td className="tabular px-2.5 py-1 text-right">{integer(column.n_unique)}</td>
                  <td className="px-2.5 py-1">
                    <span className="flex flex-wrap gap-1">
                      {column.is_constant ? <Chip>constant</Chip> : null}
                      {column.looks_like_id ? <Chip>id-like</Chip> : null}
                      {column.is_near_zero_variance ? <Chip>near-zero variance</Chip> : null}
                      {column.detected_semantic_type ? (
                        <Chip>{column.detected_semantic_type}</Chip>
                      ) : null}
                      {column.outliers && column.outliers.n_outliers > 0 ? (
                        <Chip title={`${column.outliers.method} bounds`}>
                          {integer(column.outliers.n_outliers)} outliers
                        </Chip>
                      ) : null}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Disclosure>

      {!profile.columns.length ? <p className="text-xs text-ink-3">{EMPTY}</p> : null}
    </Panel>
  );
}
