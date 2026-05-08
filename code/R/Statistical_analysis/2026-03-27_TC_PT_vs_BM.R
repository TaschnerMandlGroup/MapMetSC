# ══════════════════════════════════════════════════════════════════════════════
# BM vs PT enrichment analysis — DE timepoint, reliable units only
# Claim: marker-lo TC and early SYM-like TC are enriched in BM vs PT
# ══════════════════════════════════════════════════════════════════════════════

CLAIM_CLUSTERS <- c("CD24+ marker-lo TC", "CD24- marker-lo TC",
                    "early SYM-like TC")

# All TC clusters tested for context (not just the claimed ones)
# Allows you to see if enrichment is specific or global

# ── 1. Build the comparison table ─────────────────────────────────────────────
# One row per patient × tissue × cluster
# Restricted to: DE timepoint + reliable units (n≥30) only

enrich_df <- observed_props_agg %>%
  left_join(instability_agg %>% select(unit_id, Sample, Tissue, Timepoint,
                                       n_TC_cells, flagged, reliability),
            by = "unit_id") %>%
  filter(
    Timepoint == DE_LABEL,
    !flagged                          # reliable units only (n≥30)
  ) %>%
  select(Sample, Tissue, TC_cluster, observed_prop, n_TC_cells, reliability)

message("Reliable DE units included:")
print(count(enrich_df %>% distinct(Sample, Tissue), Tissue))

# How many patients have BOTH tissues (paired subset)?
both_tissues <- enrich_df %>%
  distinct(Sample, Tissue) %>%
  count(Sample) %>%
  filter(n == 2) %>%
  pull(Sample)

message("\nPatients with both BM and PT reliable DE units: ", length(both_tissues))
message("Patients with BM only: ",
        enrich_df %>% distinct(Sample, Tissue) %>%
          filter(Tissue == BM_LABEL) %>%
          anti_join(enrich_df %>% filter(Tissue == PT_LABEL) %>%
                      distinct(Sample), by = "Sample") %>% nrow())
message("Patients with PT only: ",
        enrich_df %>% distinct(Sample, Tissue) %>%
          filter(Tissue == PT_LABEL) %>%
          anti_join(enrich_df %>% filter(Tissue == BM_LABEL) %>%
                      distinct(Sample), by = "Sample") %>% nrow())

# ── 2. Statistical tests per cluster ─────────────────────────────────────────
# Strategy:
#   - Paired Wilcoxon for patients with both tissues (within-patient comparison)
#   - Unpaired Wilcoxon for patients with one tissue only
#   - Combined: report both, use paired where available

run_enrichment_test <- function(cluster, df, paired_samples) {
  
  bm <- df %>% filter(TC_cluster == cluster, Tissue == BM_LABEL)
  pt <- df %>% filter(TC_cluster == cluster, Tissue == PT_LABEL)
  
  # Paired subset
  bm_paired <- bm %>% filter(Sample %in% paired_samples) %>%
    arrange(Sample)
  pt_paired <- pt %>% filter(Sample %in% paired_samples) %>%
    arrange(Sample)
  
  # Unpaired test (all reliable units)
  unpaired_p <- tryCatch(
    wilcox.test(bm$observed_prop, pt$observed_prop,
                alternative = "two.sided", exact = FALSE)$p.value,
    error = function(e) NA_real_
  )
  
  # Paired test (matched patients only)
  paired_p <- tryCatch({
    if (nrow(bm_paired) >= 3 && nrow(pt_paired) >= 3 &&
        nrow(bm_paired) == nrow(pt_paired)) {
      wilcox.test(bm_paired$observed_prop, pt_paired$observed_prop,
                  paired = TRUE, alternative = "two.sided",
                  exact = FALSE)$p.value
    } else NA_real_
  }, error = function(e) NA_real_)
  
  # Fold enrichment: median BM / median PT
  med_bm  <- median(bm$observed_prop, na.rm = TRUE)
  med_pt  <- median(pt$observed_prop, na.rm = TRUE)
  fold    <- ifelse(med_pt > 0, med_bm / med_pt, NA_real_)
  
  tibble(
    TC_cluster       = cluster,
    n_BM             = nrow(bm),
    n_PT             = nrow(pt),
    n_paired         = nrow(bm_paired),
    median_BM_pct    = med_bm * 100,
    median_PT_pct    = med_pt * 100,
    delta_pp         = (med_bm - med_pt) * 100,
    fold_BM_over_PT  = fold,
    p_unpaired       = unpaired_p,
    p_paired         = paired_p
  )
}

# Run for all TC clusters
all_clusters_enrich <- map_dfr(
  tc_cols,
  run_enrichment_test,
  df             = enrich_df,
  paired_samples = both_tissues
)

# Multiple testing correction (BH) on unpaired p-values (larger n, more stable)
all_clusters_enrich <- all_clusters_enrich %>%
  mutate(
    p_adj_unpaired = p.adjust(p_unpaired, method = "BH"),
    p_adj_paired   = p.adjust(p_paired,   method = "BH"),
    direction      = case_when(
      delta_pp > 0  ~ "BM > PT",
      delta_pp < 0  ~ "PT > BM",
      TRUE          ~ "equal"
    ),
    is_claim_cluster = TC_cluster %in% CLAIM_CLUSTERS,
    # Significance tier based on unpaired (conservative)
    sig_label = case_when(
      p_adj_unpaired < 0.001 ~ "***",
      p_adj_unpaired < 0.01  ~ "**",
      p_adj_unpaired < 0.05  ~ "*",
      p_adj_unpaired < 0.10  ~ "†",
      TRUE                   ~ "ns"
    )
  ) %>%
  arrange(p_adj_unpaired)

message("\nEnrichment results for ALL clusters (sorted by adjusted p-value):")
print(all_clusters_enrich %>%
        select(TC_cluster, n_BM, n_PT, n_paired,
               median_BM_pct, median_PT_pct, delta_pp,
               fold_BM_over_PT, p_unpaired, p_adj_unpaired,
               p_paired, direction, sig_label))

message("\nResults for CLAIM clusters specifically:")
print(all_clusters_enrich %>%
        filter(is_claim_cluster) %>%
        select(TC_cluster, median_BM_pct, median_PT_pct, delta_pp,
               fold_BM_over_PT, p_adj_unpaired, p_paired, sig_label))

# ── 3. Plot A — paired dot plot for claim clusters ────────────────────────────
# Shows within-patient BM vs PT for paired patients — strongest visual
# for the metastasis claim

paired_long <- enrich_df %>%
  filter(Sample %in% both_tissues,
         TC_cluster %in% CLAIM_CLUSTERS) %>%
  mutate(
    TC_cluster = factor(TC_cluster, levels = CLAIM_CLUSTERS),
    Tissue     = factor(Tissue, levels = c(PT_LABEL, BM_LABEL))
  )

# Median summary for overlay
paired_summary <- paired_long %>%
  group_by(TC_cluster, Tissue) %>%
  summarise(median_prop = median(observed_prop), .groups = "drop")

# Significance labels for paired test
sig_df <- all_clusters_enrich %>%
  filter(is_claim_cluster) %>%
  mutate(
    TC_cluster = factor(TC_cluster, levels = CLAIM_CLUSTERS),
    sig_paired = case_when(
      p_paired < 0.001 ~ "***",
      p_paired < 0.01  ~ "**",
      p_paired < 0.05  ~ "*",
      p_paired < 0.10  ~ "†",
      TRUE             ~ "ns"
    ),
    sig_unpaired = case_when(
      p_adj_unpaired < 0.001 ~ "***",
      p_adj_unpaired < 0.01  ~ "**",
      p_adj_unpaired < 0.05  ~ "*",
      p_adj_unpaired < 0.10  ~ "†",
      TRUE                   ~ "ns"
    ),
    label = paste0("paired: ", sig_paired,
                   "\nunpaired: ", sig_unpaired,
                   "\nΔ=", round(delta_pp, 1), "pp",
                   "\n", round(fold_BM_over_PT, 1), "× BM/PT")
  )

p_paired_dot <- ggplot(paired_long,
                       aes(x = Tissue, y = observed_prop * 100)) +
  
  # Lines connecting paired patients
  geom_line(aes(group = Sample),
            colour = "grey70", linewidth = 0.4, alpha = 0.6) +
  
  # Individual points
  geom_point(aes(colour = Tissue), size = 2.5, alpha = 0.8) +
  
  # Median crossbar
  geom_crossbar(data    = paired_summary,
                aes(y   = median_prop * 100,
                    ymin = median_prop * 100,
                    ymax = median_prop * 100,
                    colour = Tissue),
                width = 0.4, linewidth = 1) +
  
  # Significance annotation
  geom_text(data    = sig_df,
            aes(x   = 1.5, y = Inf, label = label),
            vjust   = 1.3, size = 2.8,
            colour  = "grey25", inherit.aes = FALSE) +
  
  facet_wrap(~ TC_cluster, scales = "free_y", ncol = 3) +
  
  scale_colour_manual(
    values = c("BM" = "#E24B4A", "PT" = "#185FA5"),
    name   = NULL
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    title    = "BM vs PT proportion — claim clusters, DE timepoint, reliable units",
    subtitle = paste0(
      "Lines = paired patients (n=", length(both_tissues), ") | ",
      "crossbar = median | paired Wilcoxon + unpaired BH-adjusted Wilcoxon\n",
      "Claim: marker-lo TC and early SYM-like TC are enriched in BM vs PT"
    ),
    x = NULL,
    y = "TC cluster proportion (%)"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title       = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle    = element_text(size = 8.5, colour = "grey35"),
    strip.text       = element_text(face = "bold", size = 10),
    strip.background = element_blank(),
    axis.text        = element_text(colour = "black"),
    legend.position  = "none"
  )

# ── 4. Plot B — forest plot of ALL clusters (BM vs PT delta) ─────────────────
# Shows whether enrichment is specific to claim clusters or global

forest_order <- all_clusters_enrich %>%
  arrange(delta_pp) %>%
  pull(TC_cluster)

p_forest <- all_clusters_enrich %>%
  mutate(
    TC_cluster   = factor(TC_cluster, levels = forest_order),
    point_colour = case_when(
      is_claim_cluster & delta_pp > 0 & p_adj_unpaired < 0.05 ~ "claim — significant",
      is_claim_cluster                                          ~ "claim — ns",
      delta_pp > 0 & p_adj_unpaired < 0.05                    ~ "other — significant",
      TRUE                                                      ~ "other — ns"
    ) %>% factor(levels = c("claim — significant", "claim — ns",
                            "other — significant", "other — ns"))
  ) %>%
  ggplot(aes(y = TC_cluster, x = delta_pp, colour = point_colour)) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.5) +
  geom_segment(aes(x = 0, xend = delta_pp, yend = TC_cluster),
               linewidth = 1.0, alpha = 0.7) +
  geom_point(size = 4) +
  geom_text(aes(x      = delta_pp,
                label  = paste0(sig_label, "  ",
                                round(median_BM_pct, 1), "% vs ",
                                round(median_PT_pct, 1), "%")),
            hjust  = ifelse(all_clusters_enrich %>%
                              arrange(delta_pp) %>%
                              pull(delta_pp) >= 0, -0.1, 1.1),
            size   = 2.8, colour = "grey25") +
  scale_colour_manual(
    values = c("claim — significant" = "#E24B4A",
               "claim — ns"          = "#F0997B",
               "other — significant" = "#185FA5",
               "other — ns"          = "#B5D4F4"),
    name   = NULL
  ) +
  scale_x_continuous(labels = function(x) paste0(x, " pp"),
                     expand = expansion(mult = 0.3)) +
  labs(
    title    = "BM vs PT enrichment — all TC clusters, DE timepoint, reliable units",
    subtitle = "Δ = median BM% − median PT% | label: sig + median BM% vs PT%\n† p<0.10, * p<0.05, ** p<0.01, *** p<0.001 (BH-adjusted unpaired Wilcoxon)",
    x        = "Median proportion difference (BM − PT, percentage points)",
    y        = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 8.5, colour = "grey35"),
    axis.text       = element_text(colour = "black"),
    legend.position = "right",
    legend.key.size = unit(0.4, "cm")
  )

# ── 5. Plot C — violin + jitter for all reliable DE units (full distribution) ─
violin_order <- all_clusters_enrich %>%
  arrange(desc(delta_pp)) %>%
  pull(TC_cluster)

p_violin <- enrich_df %>%
  filter(TC_cluster %in% tc_cols) %>%
  mutate(
    TC_cluster      = factor(TC_cluster, levels = violin_order),
    Tissue          = factor(Tissue, levels = c(PT_LABEL, BM_LABEL)),
    is_claim        = TC_cluster %in% CLAIM_CLUSTERS
  ) %>%
  ggplot(aes(x = Tissue, y = observed_prop * 100, fill = Tissue)) +
  geom_violin(linewidth = 0.3, alpha = 0.5, colour = "grey60") +
  geom_jitter(aes(colour = Tissue), width = 0.15, size = 1.2,
              alpha = 0.7) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.3,
               colour = "grey30", linewidth = 0.4) +
  # Highlight claim clusters with bold strip text
  ggh4x::facet_wrap2(
    ~ TC_cluster, scales = "free_y", ncol = 5,
    strip = ggh4x::strip_themed(
      background_x = lapply(violin_order, function(cl) {
        if (cl %in% CLAIM_CLUSTERS)
          element_rect(fill = "#FAECE7", colour = "#D85A30", linewidth = 0.8)
        else
          element_rect(fill = "grey96", colour = "grey80", linewidth = 0.4)
      })
    )
  ) +
  scale_fill_manual(values   = c("BM" = "#E24B4A", "PT" = "#185FA5"),
                    name     = NULL) +
  scale_colour_manual(values = c("BM" = "#E24B4A", "PT" = "#185FA5"),
                      guide  = "none") +
  scale_y_continuous(labels  = function(x) paste0(x, "%")) +
  labs(
    title    = "TC cluster proportion distributions — BM vs PT, DE timepoint",
    subtitle = "Highlighted panels = claim clusters | reliable units only (n≥30 TC cells)",
    x        = NULL,
    y        = "TC cluster proportion (%)"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title       = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle    = element_text(size = 8.5, colour = "grey35"),
    strip.text       = element_text(face = "bold", size = 8),
    axis.text        = element_text(colour = "black"),
    legend.position  = "top",
    legend.key.size  = unit(0.4, "cm")
  )

print(p_paired_dot)
print(p_forest)


if (requireNamespace("ggh4x", quietly = TRUE)) {
  print(p_violin)
} else {
  message("ggh4x not installed — using standard facet_wrap for violin plot")
  p_violin_simple <- enrich_df %>%
    filter(TC_cluster %in% tc_cols) %>%
    mutate(
      TC_cluster = factor(TC_cluster, levels = violin_order),
      Tissue     = factor(Tissue, levels = c(PT_LABEL, BM_LABEL))
    ) %>%
    ggplot(aes(x = Tissue, y = observed_prop * 100, fill = Tissue)) +
    geom_violin(linewidth = 0.3, alpha = 0.5, colour = "grey60") +
    geom_jitter(aes(colour = Tissue), width = 0.15, size = 1.2, alpha = 0.7) +
    geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.3,
                 colour = "grey30", linewidth = 0.4) +
    facet_wrap(~ TC_cluster, scales = "free_y", ncol = 5) +
    scale_fill_manual(values   = c("BM" = "#E24B4A", "PT" = "#185FA5"),
                      name     = NULL) +
    scale_colour_manual(values = c("BM" = "#E24B4A", "PT" = "#185FA5"),
                        guide  = "none") +
    scale_y_continuous(labels  = function(x) paste0(x, "%")) +
    labs(
      title    = "TC cluster proportion distributions — BM vs PT, DE timepoint",
      subtitle = "Reliable units only (n≥30 TC cells) | claim clusters: CD24+/- marker-lo, early SYM-like",
      x        = NULL,
      y        = "TC cluster proportion (%)"
    ) +
    theme_classic(base_size = 10) +
    theme(
      plot.title       = element_text(size = 12, face = "bold", colour = "black"),
      plot.subtitle    = element_text(size = 8.5, colour = "grey35"),
      strip.text       = element_text(face = "bold", size = 8),
      axis.text        = element_text(colour = "black"),
      legend.position  = "top",
      legend.key.size  = unit(0.4, "cm")
    )
  print(p_violin_simple)
  ggsave(paste0(out,"15_violin_all_clusters_BM_PT.pdf"),  
         p_violin_simple,width = 12,  height = 8,   useDingbats = FALSE)
}

# ── 6. Export ─────────────────────────────────────────────────────────────────
all_clusters_enrich %>%
  select(TC_cluster, n_BM, n_PT, n_paired,
         median_BM_pct, median_PT_pct, delta_pp, fold_BM_over_PT,
         p_unpaired, p_adj_unpaired, p_paired, p_adj_paired,
         direction, sig_label, is_claim_cluster) %>%
  write_csv("BM_vs_PT_enrichment_DE_reliable.csv")

ggsave(paste0(out,"13_paired_dot_claim_clusters.pdf"),  p_paired_dot,   width = 9,   height = 4.5, useDingbats = FALSE)
ggsave(paste0(out,"14_forest_all_clusters_BM_PT.pdf"),  p_forest,       width = 9,   height = 5,   useDingbats = FALSE)

