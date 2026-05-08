library(tidyverse)
library(ggrepel)
library(patchwork)

set.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0. Parameters — adjust all values in this block only
# ══════════════════════════════════════════════════════════════════════════════

TC_PATTERN       <- "TC|[Tt]umor"  # regex matching TC celltypes in seu_meta$celltype
BM_LABEL         <- "BM"           # exact string in metadata$Tissue for bone marrow
DE_LABEL         <- "DE"           # exact string in metadata$Timepoint for DE

N_BOOT           <- 1000           # bootstrap iterations per unit
MIN_TC_CELLS     <- 30             # minimum pooled TC cells for reliable calls
MAX_MEAN_SD      <- 0.10           # flag if mean bootstrap SD > this value

MIN_DETECT_CELLS <- 3              # minimum absolute cells for cluster detection
MIN_DETECT_PROP  <- 0.01           # minimum proportion for cluster detection (1%)

# extract info from metadata
seu_meta_annot <- seu_meta %>%
  left_join(
    metadata %>%
      select(Sample, Tissue, Timepoint) %>%
      distinct(Sample, Tissue, .keep_all = TRUE),
    by = c("Sample" = "Sample",
           "tissue" = "Tissue")
  ) %>%
  rename(Tissue = tissue)    # <── restore capital-T column name for downstream code

# Verify
stopifnot(nrow(seu_meta_annot) == nrow(seu_meta))
message("✓ Columns present: ", paste(colnames(seu_meta_annot), collapse = ", "))

missing_tp <- seu_meta_annot %>%
  filter(is.na(Timepoint)) %>%
  distinct(Sample, Tissue)
if (nrow(missing_tp) > 0) {
  warning("These Sample × Tissue combos have no Timepoint:")
  print(missing_tp)
} else {
  message("✓ All cells have Timepoint annotation")
}

# Verify row count preserved exactly
stopifnot(nrow(seu_meta_annot) == nrow(seu_meta))
message("✓ seu_meta_annot rows: ", nrow(seu_meta_annot),
        " | matches expected: ", nrow(seu_meta))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Identify TC clusters
# ══════════════════════════════════════════════════════════════════════════════

tc_cols <- grep(TC_PATTERN, unique(seu_meta_annot$celltype), value = TRUE)
message("TC clusters found: ", paste(tc_cols, collapse = ", "))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Define aggregation unit: patient × tissue × timepoint
#    All ROIs for the same (Sample, Tissue, Timepoint) are pooled
# ══════════════════════════════════════════════════════════════════════════════

tc_meta_agg <- seu_meta_annot %>%
  filter(grepl(TC_PATTERN, celltype)) %>%
  select(Sample, Tissue, Timepoint, sample_id, TC_cluster = celltype) %>%
  mutate(unit_id = paste0(Sample, " | ", Tissue, " | ", Timepoint))

# ROIs pooled per unit
rois_per_unit <- seu_meta_annot %>%
  mutate(unit_id = paste0(Sample, " | ", Tissue, " | ", Timepoint)) %>%
  distinct(unit_id, sample_id) %>%
  count(unit_id, name = "n_ROIs")

# Per-unit TC cell counts (total pooled)
tc_counts_agg <- tc_meta_agg %>%
  count(unit_id, name = "n_TC_cells")

# Per-unit × per-cluster absolute cell counts (for detection threshold)
tc_counts_per_cluster <- tc_meta_agg %>%
  count(unit_id, TC_cluster, name = "n_cluster_cells")

# Per-unit observed proportions
observed_props_agg <- tc_meta_agg %>%
  count(unit_id, TC_cluster) %>%
  group_by(unit_id) %>%
  mutate(observed_prop = n / sum(n)) %>%
  ungroup()

message("\nUnits per tissue × timepoint combination:")
tc_meta_agg %>%
  distinct(unit_id, Sample, Tissue, Timepoint) %>%
  count(Tissue, Timepoint) %>%
  print()

# ══════════════════════════════════════════════════════════════════════════════
# 5. Bootstrap at patient × tissue × timepoint level
# ══════════════════════════════════════════════════════════════════════════════

run_bootstrap_unit <- function(uid, tc_data, n_boot, all_clusters) {
  cells <- tc_data %>% filter(unit_id == uid) %>% pull(TC_cluster)
  n     <- length(cells)
  
  resamples <- replicate(n_boot, {
    s     <- sample(cells, size = n, replace = TRUE)
    props <- table(factor(s, levels = all_clusters)) / n
    as.numeric(props)
  })
  
  tibble(
    unit_id    = uid,
    TC_cluster = all_clusters,
    boot_mean  = rowMeans(resamples),
    boot_sd    = apply(resamples, 1, sd),
    boot_q025  = apply(resamples, 1, quantile, 0.025),
    boot_q975  = apply(resamples, 1, quantile, 0.975),
    boot_q25   = apply(resamples, 1, quantile, 0.25),
    boot_q75   = apply(resamples, 1, quantile, 0.75),
    n_TC_cells = n
  )
}

message("Bootstrapping ", n_distinct(tc_meta_agg$unit_id),
        " units × ", N_BOOT, " iterations...")

boot_results_agg <- map_dfr(
  unique(tc_meta_agg$unit_id),
  run_bootstrap_unit,
  tc_data      = tc_meta_agg,
  n_boot       = N_BOOT,
  all_clusters = tc_cols
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Instability scores per unit
# ══════════════════════════════════════════════════════════════════════════════

instability_agg <- boot_results_agg %>%
  group_by(unit_id, n_TC_cells) %>%
  summarise(
    mean_boot_SD  = mean(boot_sd),
    mean_CI_width = mean(boot_q975 - boot_q025),
    max_CI_width  = max(boot_q975 - boot_q025),
    .groups = "drop"
  ) %>%
  separate(unit_id, into = c("Sample", "Tissue", "Timepoint"),
           sep = " \\| ", remove = FALSE) %>%
  left_join(rois_per_unit, by = "unit_id") %>%
  mutate(
    flag_n      = n_TC_cells < MIN_TC_CELLS,
    flag_sd     = mean_boot_SD > MAX_MEAN_SD,
    flagged     = flag_n | flag_sd,
    flag_reason = case_when(
      flag_n & flag_sd ~ paste0("n=", n_TC_cells, " & SD>", MAX_MEAN_SD),
      flag_n           ~ paste0("n=", n_TC_cells, " < ", MIN_TC_CELLS),
      flag_sd          ~ paste0("mean SD=", round(mean_boot_SD, 3)),
      TRUE             ~ "OK"
    ),
    reliability = case_when(
      n_TC_cells <  10 ~ "critical (n<10)",
      n_TC_cells <  30 ~ "unreliable (n<30)",
      n_TC_cells < 100 ~ "marginal (n<100)",
      TRUE             ~ "reliable (n≥100)"
    ) %>% factor(levels = c("critical (n<10)", "unreliable (n<30)",
                            "marginal (n<100)", "reliable (n≥100)")),
    is_BM    = Tissue    == BM_LABEL,
    is_DE    = Timepoint == DE_LABEL,
    is_BM_DE = is_BM & is_DE,
    group    = case_when(
      is_BM_DE          ~ "BM + DE",
      is_BM & !is_DE    ~ "BM, other timepoint",
      !is_BM & is_DE    ~ "Tumor + DE",
      TRUE              ~ "Tumor, other timepoint"
    ) %>% factor(levels = c("BM + DE", "BM, other timepoint",
                            "Tumor + DE", "Tumor, other timepoint"))
  )

# Colour palette
rel_colours <- c(
  "critical (n<10)"   = "#E24B4A",
  "unreliable (n<30)" = "#EF9F27",
  "marginal (n<100)"  = "#BA7517",
  "reliable (n≥100)"  = "#185FA5"
)

message("\nUnit counts by group:")
print(count(instability_agg, group))

message("\nBM + DE units — reliability summary:")
instability_agg %>%
  filter(is_BM_DE) %>%
  arrange(n_TC_cells) %>%
  select(unit_id, n_ROIs, n_TC_cells, mean_boot_SD,
         mean_CI_width, reliability, flag_reason) %>%
  print(n = Inf)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Plot A — full instability plot, BM+DE highlighted
# ══════════════════════════════════════════════════════════════════════════════

bm_de_units <- instability_agg %>% filter(is_BM_DE)

p_highlighted <- ggplot(instability_agg,
                        aes(x = n_TC_cells, y = mean_boot_SD,
                            colour = reliability)) +
  geom_vline(xintercept = MIN_TC_CELLS, linetype = "dashed",
             colour = "grey60", linewidth = 0.5) +
  geom_hline(yintercept = MAX_MEAN_SD, linetype = "dashed",
             colour = "grey60", linewidth = 0.5) +
  geom_point(aes(size = n_ROIs), alpha = 0.72) +
  geom_point(data   = bm_de_units,
             aes(size = n_ROIs),
             shape  = 21, stroke = 1.2,
             colour = "black", fill = NA) +
  geom_text_repel(
    data           = bm_de_units,
    aes(label      = unit_id),
    size           = 2.5, fontface = "bold",
    max.overlaps   = 30, box.padding = 0.4,
    segment.size   = 0.3, segment.colour = "grey50",
    colour         = "black"
  ) +
  annotate("text", x = MIN_TC_CELLS, y = Inf,
           label = paste0("min n = ", MIN_TC_CELLS),
           hjust = -0.1, vjust = 1.5, size = 3, colour = "grey45") +
  annotate("text", x = Inf, y = MAX_MEAN_SD,
           label = paste0("SD threshold = ", MAX_MEAN_SD),
           hjust = 1.05, vjust = -0.5, size = 3, colour = "grey45") +
  scale_colour_manual(values = rel_colours, name = "Reliability") +
  scale_size_continuous(range = c(2, 7), name = "ROIs pooled") +
  scale_x_continuous(trans  = "log10",
                     breaks = c(1, 5, 10, 30, 100, 500, 1000),
                     labels = scales::comma) +
  labs(
    title    = "TC cluster call stability — BM+DE units highlighted",
    subtitle = paste0(sum(bm_de_units$flagged), " of ", nrow(bm_de_units),
                      " BM+DE units flagged | one dot = patient × tissue × timepoint",
                      " | dot size = ROIs pooled"),
    x        = "TC cells pooled across ROIs (log scale)",
    y        = "Mean bootstrap SD (proportion units)"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 9,  colour = "grey35"),
    axis.text       = element_text(colour = "black"),
    legend.position = "right",
    legend.key.size = unit(0.4, "cm")
  )

print(p_highlighted)

# ══════════════════════════════════════════════════════════════════════════════
# 8. Plot B — BM+DE ribbon plot per TC cluster
# ══════════════════════════════════════════════════════════════════════════════

bm_de_unit_ids <- bm_de_units %>%
  arrange(n_TC_cells) %>%
  pull(unit_id)

plot_df_bm_de <- boot_results_agg %>%
  filter(unit_id %in% bm_de_unit_ids) %>%
  select(-n_TC_cells) %>%                              # drop before join to avoid collision
  left_join(observed_props_agg %>% select(unit_id, TC_cluster, observed_prop),
            by = c("unit_id", "TC_cluster")) %>%
  left_join(instability_agg %>% select(unit_id, flagged, reliability,
                                       n_ROIs, n_TC_cells),
            by = "unit_id") %>%
  replace_na(list(observed_prop = 0)) %>%
  mutate(
    unit_label = paste0(ifelse(flagged, "⚑ ", ""), unit_id,
                        "\n(", n_ROIs, " ROIs, n=", n_TC_cells, ")"),
    unit_label = factor(unit_label,
                        levels = unique(unit_label[order(
                          match(unit_id, bm_de_unit_ids))]))
  )

p_bm_de_ribbon <- ggplot(plot_df_bm_de,
                         aes(x      = unit_label,
                             colour = reliability,
                             fill   = reliability)) +
  geom_linerange(aes(ymin = boot_q025, ymax = boot_q975),
                 linewidth = 3.5, alpha = 0.18) +
  geom_linerange(aes(ymin = boot_q25,  ymax = boot_q75),
                 linewidth = 3.5, alpha = 0.45) +
  geom_point(aes(y = observed_prop), size = 2.2, alpha = 0.95) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3) +
  facet_wrap(~ TC_cluster, scales = "free_y", ncol = 2) +
  scale_colour_manual(values = rel_colours, name = "Reliability") +
  scale_fill_manual(values   = rel_colours, name = "Reliability") +
  scale_y_continuous(labels  = scales::percent_format(accuracy = 1)) +
  labs(
    title    = paste0("Bootstrap stability — BM units at ", DE_LABEL, " timepoint"),
    subtitle = "Point = observed; thick = IQR; thin = 95% CI | ⚑ = flagged | sorted low → high pooled TC cells\nLabel: patient | tissue | timepoint (ROIs pooled, total TC cells)",
    x        = NULL,
    y        = "TC cluster proportion"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title       = element_text(size = 12, face = "bold",  colour = "black"),
    plot.subtitle    = element_text(size = 8,  colour = "grey35"),
    axis.text.x      = element_text(angle = 45, hjust = 1, size = 7,
                                    colour = "black"),
    strip.text       = element_text(face = "bold", size = 9),
    strip.background = element_blank(),
    legend.position  = "top",
    legend.key.size  = unit(0.4, "cm")
  )

print(p_bm_de_ribbon)

# ══════════════════════════════════════════════════════════════════════════════
# 9. Plot C — CI width bar chart, BM+DE units
# ══════════════════════════════════════════════════════════════════════════════

# Pre-compute label order before entering ggplot (avoids fct_reorder ambiguity)
p_table_order <- instability_agg %>%
  filter(is_BM_DE) %>%
  arrange(n_TC_cells) %>%
  mutate(unit_label = paste0(unit_id, " (", n_ROIs, " ROIs)")) %>%
  pull(unit_label)

p_table <- instability_agg %>%
  filter(is_BM_DE) %>%
  arrange(n_TC_cells) %>%
  mutate(
    unit_label = paste0(unit_id, " (", n_ROIs, " ROIs)"),
    unit_label = factor(unit_label, levels = p_table_order)
  ) %>%
  ggplot(aes(y = unit_label)) +
  geom_col(aes(x = mean_CI_width, fill = reliability),
           width = 0.7, alpha = 0.85) +
  geom_vline(xintercept = 0.40, linetype = "dashed",
             colour = "grey50", linewidth = 0.5) +
  geom_text(aes(x     = mean_CI_width + 0.008,
                label = paste0("n=", n_TC_cells, " cells  ", flag_reason)),
            hjust = 0, size = 2.8, colour = "grey25") +
  scale_fill_manual(values = rel_colours, name = "Reliability") +
  scale_x_continuous(limits = c(0, 1.2),
                     labels = scales::percent_format(accuracy = 1)) +
  labs(
    title    = "BM + DE units: mean bootstrap 95% CI width",
    subtitle = "Dashed = 40 pp threshold | sorted low → high pooled TC cells",
    x        = "Mean 95% CI width across TC clusters",
    y        = NULL
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 8.5, colour = "grey35"),
    axis.text       = element_text(colour = "black"),
    legend.position = "right",
    legend.key.size = unit(0.4, "cm")
  )

print(p_table)

# ══════════════════════════════════════════════════════════════════════════════
# 10. Test 1 — Per-cluster instability score across all units
# ══════════════════════════════════════════════════════════════════════════════

cluster_instability <- boot_results_agg %>%
  select(-n_TC_cells) %>%                              # drop before join
  left_join(instability_agg %>% select(unit_id, reliability, flagged,
                                       n_TC_cells),
            by = "unit_id") %>%
  group_by(TC_cluster, reliability) %>%
  summarise(
    mean_SD       = mean(boot_sd),
    median_SD     = median(boot_sd),
    mean_CI_width = mean(boot_q975 - boot_q025),
    n_units       = n(),
    .groups = "drop"
  )

cluster_rank <- boot_results_agg %>%
  select(-n_TC_cells) %>%                              # drop before join
  left_join(instability_agg %>% select(unit_id, n_TC_cells), by = "unit_id") %>%
  group_by(TC_cluster) %>%
  summarise(
    overall_mean_SD   = mean(boot_sd),
    overall_median_SD = median(boot_sd),
    .groups = "drop"
  ) %>%
  arrange(desc(overall_mean_SD)) %>%
  mutate(instability_rank = row_number())

message("\nCluster instability ranking (most → least unstable):")
print(cluster_rank)

t1_cluster_order <- cluster_rank %>%
  arrange(overall_mean_SD) %>%
  pull(TC_cluster)

# Plot T1A: stacked mean SD by reliability tier
p_cluster_sd <- cluster_instability %>%
  left_join(cluster_rank %>% select(TC_cluster, instability_rank),
            by = "TC_cluster") %>%
  mutate(TC_cluster = factor(TC_cluster, levels = t1_cluster_order)) %>%
  ggplot(aes(x = TC_cluster, y = mean_SD, fill = reliability)) +
  geom_col(position = "stack", width = 0.7, alpha = 0.88) +
  geom_hline(yintercept = MAX_MEAN_SD, linetype = "dashed",
             colour = "grey50", linewidth = 0.5) +
  annotate("text", x = Inf, y = MAX_MEAN_SD,
           label = paste0("SD threshold = ", MAX_MEAN_SD),
           hjust = 1.05, vjust = -0.5, size = 3, colour = "grey45") +
  scale_fill_manual(values = rel_colours, name = "Reliability tier") +
  labs(
    title    = "Test 1 — Per-cluster bootstrap instability across all units",
    subtitle = "Stacked by reliability tier | sorted most → least unstable\nHigh SD only in unreliable units = likely artifact; high SD in reliable units = variable biology",
    x        = NULL,
    y        = "Mean bootstrap SD (proportion units)"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 8.5, colour = "grey35"),
    axis.text.x     = element_text(angle = 35, hjust = 1, colour = "black"),
    axis.text.y     = element_text(colour = "black"),
    legend.position = "right",
    legend.key.size = unit(0.4, "cm")
  )

# Plot T1B: violin + jitter
p_cluster_violin <- boot_results_agg %>%
  select(-n_TC_cells) %>%                              # drop before join
  left_join(instability_agg %>% select(unit_id, reliability, n_TC_cells),
            by = "unit_id") %>%
  mutate(TC_cluster = factor(TC_cluster, levels = t1_cluster_order)) %>%
  ggplot(aes(x = TC_cluster, y = boot_sd, colour = reliability)) +
  geom_violin(aes(group = TC_cluster), fill = "grey92", colour = "grey70",
              linewidth = 0.4, alpha = 0.6) +
  geom_jitter(width = 0.2, size = 1.4, alpha = 0.65) +
  geom_hline(yintercept = MAX_MEAN_SD, linetype = "dashed",
             colour = "grey50", linewidth = 0.5) +
  scale_colour_manual(values = rel_colours, name = "Reliability") +
  labs(
    title    = "Test 1b — Bootstrap SD distribution per cluster across all units",
    subtitle = "Each dot = one patient×tissue×timepoint unit | violin = distribution shape",
    x        = NULL,
    y        = "Bootstrap SD (proportion units)"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 8.5, colour = "grey35"),
    axis.text.x     = element_text(angle = 35, hjust = 1, colour = "black"),
    axis.text.y     = element_text(colour = "black"),
    legend.position = "right",
    legend.key.size = unit(0.4, "cm")
  )

print(p_cluster_sd)
print(p_cluster_violin)

# ══════════════════════════════════════════════════════════════════════════════
# 11. Test 2 — Detection rate analysis (strict threshold)
# ══════════════════════════════════════════════════════════════════════════════

detection_df <- observed_props_agg %>%
  left_join(tc_counts_per_cluster,
            by = c("unit_id", "TC_cluster")) %>%
  replace_na(list(n_cluster_cells = 0)) %>%
  left_join(instability_agg %>% select(unit_id, n_TC_cells, flagged,
                                       reliability),
            by = "unit_id") %>%
  mutate(
    detected = n_cluster_cells >= MIN_DETECT_CELLS &
      observed_prop   >= MIN_DETECT_PROP,
    n_bin    = cut(n_TC_cells,
                   breaks         = c(0, 10, 30, 100, 500, Inf),
                   labels         = c("<10", "10–30", "30–100", "100–500", ">500"),
                   right          = TRUE,
                   include.lowest = TRUE)
  )

message("\nDetection threshold summary (≥", MIN_DETECT_CELLS,
        " cells AND ≥", MIN_DETECT_PROP * 100, "% proportion):")
detection_df %>%
  count(detected) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()

detection_rate <- detection_df %>%
  group_by(TC_cluster, n_bin, .drop = FALSE) %>%
  summarise(
    n_units        = n(),
    n_detected     = sum(detected),
    detection_rate = n_detected / n_units,
    .groups = "drop"
  )

artifact_cor <- detection_df %>%
  group_by(TC_cluster) %>%
  summarise(
    spearman_r   = cor(n_TC_cells, as.numeric(detected),
                       method = "spearman", use = "complete.obs"),
    p_value      = tryCatch(
      cor.test(n_TC_cells, as.numeric(detected),
               method = "spearman", exact = FALSE)$p.value,
      error = function(e) NA_real_
    ),
    pct_detected_flagged       = mean(detected[flagged == TRUE],  na.rm = TRUE) * 100,
    pct_detected_reliable      = mean(detected[flagged == FALSE], na.rm = TRUE) * 100,
    enrichment_in_flagged      = pct_detected_flagged - pct_detected_reliable,
    median_cells_when_detected = median(n_cluster_cells[detected], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    p_adj         = p.adjust(p_value, method = "BH"),
    artifact_flag = spearman_r < 0 & p_adj < 0.05,
    artifact_label = case_when(
      spearman_r < -0.3 & p_adj < 0.05  ~ "likely artifact",
      spearman_r < 0    & p_adj < 0.05  ~ "possible artifact",
      spearman_r < 0    & p_adj >= 0.05 ~ "inconclusive",
      TRUE                               ~ "likely biology"
    ) %>% factor(levels = c("likely artifact", "possible artifact",
                            "inconclusive",    "likely biology"))
  ) %>%
  arrange(spearman_r)

message("\nDetection rate artifact scores (most artifact-like first):")
print(artifact_cor %>%
        select(TC_cluster, spearman_r, p_adj, pct_detected_flagged,
               pct_detected_reliable, enrichment_in_flagged,
               median_cells_when_detected, artifact_label))

artifact_colours <- c(
  "likely artifact"    = "#E24B4A",
  "possible artifact"  = "#EF9F27",
  "inconclusive"       = "#888780",
  "likely biology"     = "#185FA5"
)

# Pre-compute cluster orders — one value per cluster, no ambiguity
t2_heat_order   <- artifact_cor %>% arrange(spearman_r)            %>% pull(TC_cluster)
t2_enrich_order <- artifact_cor %>% arrange(enrichment_in_flagged) %>% pull(TC_cluster)

# Plot T2A: detection rate heatmap
p_detection_heat <- detection_rate %>%
  left_join(artifact_cor %>% select(TC_cluster, artifact_label, spearman_r),
            by = "TC_cluster") %>%
  mutate(TC_cluster = factor(TC_cluster, levels = t2_heat_order)) %>%
  ggplot(aes(x = n_bin, y = TC_cluster, fill = detection_rate)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = paste0(n_detected, "/", n_units)),
            size = 2.8, colour = "white") +
  scale_fill_gradientn(
    colours = c("#E6F1FB", "#378ADD", "#042C53"),
    limits  = c(0, 1),
    labels  = scales::percent_format(accuracy = 1),
    name    = "Detection\nrate"
  ) +
  labs(
    title    = "Test 2 — Cluster detection rate by n_TC_cells bin",
    subtitle = paste0(
      "Detected = ≥", MIN_DETECT_CELLS, " cells AND ≥",
      MIN_DETECT_PROP * 100, "% proportion | cell text = detected/total units\n",
      "Top rows = artifact-like (only detected in low-n samples)"
    ),
    x = "TC cells in unit (pooled across ROIs)",
    y = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title        = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle     = element_text(size = 8.5, colour = "grey35"),
    axis.text         = element_text(colour = "black"),
    legend.key.height = unit(0.8, "cm")
  )

# Plot T2B: enrichment in flagged vs reliable
# hjust computed inside data to avoid vector-length mismatch
p_enrichment <- artifact_cor %>%
  mutate(
    TC_cluster = factor(TC_cluster, levels = t2_enrich_order),
    hjust_val  = ifelse(enrichment_in_flagged > 0, -0.15, 1.15)
  ) %>%
  ggplot(aes(y = TC_cluster, colour = artifact_label)) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.5) +
  geom_segment(aes(x = 0, xend = enrichment_in_flagged, yend = TC_cluster),
               linewidth = 1.2, alpha = 0.8) +
  geom_point(aes(x = enrichment_in_flagged), size = 3.5) +
  geom_text(
    aes(x     = enrichment_in_flagged,
        label = paste0(round(pct_detected_flagged, 0), "% vs ",
                       round(pct_detected_reliable, 0), "%"),
        hjust = hjust_val),
    size   = 2.8,
    colour = "grey25"
  ) +
  scale_colour_manual(values = artifact_colours, name = "Artifact assessment") +
  scale_x_continuous(labels = function(x) paste0(x, " pp"),
                     expand = expansion(mult = 0.25)) +
  labs(
    title    = "Test 2b — Detection enrichment in flagged vs reliable units",
    subtitle = paste0(
      "Detection threshold: ≥", MIN_DETECT_CELLS,
      " cells AND ≥", MIN_DETECT_PROP * 100, "% proportion\n",
      "Positive = cluster detected more often in low-n (flagged) samples = artifact signal"
    ),
    x = "Detection rate difference (flagged − reliable, percentage points)",
    y = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 8.5, colour = "grey35"),
    axis.text       = element_text(colour = "black"),
    legend.position = "right",
    legend.key.size = unit(0.4, "cm")
  )

print(p_detection_heat)
print(p_enrichment)

# ══════════════════════════════════════════════════════════════════════════════
# 12. Test 3 — Zero-overlap test
# ══════════════════════════════════════════════════════════════════════════════

zero_overlap <- boot_results_agg %>%
  select(-n_TC_cells) %>%                              # drop before join
  left_join(observed_props_agg %>% select(unit_id, TC_cluster, observed_prop),
            by = c("unit_id", "TC_cluster")) %>%
  left_join(tc_counts_per_cluster,
            by = c("unit_id", "TC_cluster")) %>%
  replace_na(list(observed_prop = 0, n_cluster_cells = 0)) %>%
  left_join(instability_agg %>% select(unit_id, n_TC_cells, flagged,
                                       reliability),
            by = "unit_id") %>%
  mutate(
    CI_overlaps_zero = boot_q025 <= 0,
    detected         = n_cluster_cells >= MIN_DETECT_CELLS &
      observed_prop   >= MIN_DETECT_PROP
  )

zero_summary <- zero_overlap %>%
  filter(detected) %>%
  group_by(TC_cluster) %>%
  summarise(
    n_detected             = n(),
    n_CI_overlaps_zero     = sum(CI_overlaps_zero),
    pct_CI_overlaps_zero   = mean(CI_overlaps_zero) * 100,
    n_flagged_detected     = sum(flagged),
    n_flagged_overlap_zero = sum(CI_overlaps_zero & flagged),
    pct_flagged_overlap    = mean(CI_overlaps_zero[flagged],  na.rm = TRUE) * 100,
    pct_reliable_overlap   = mean(CI_overlaps_zero[!flagged], na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  left_join(artifact_cor %>% select(TC_cluster, artifact_label, spearman_r),
            by = "TC_cluster") %>%
  arrange(desc(pct_CI_overlaps_zero))

message("\nZero-overlap test results (most overlapping first):")
print(zero_summary %>%
        select(TC_cluster, n_detected, pct_CI_overlaps_zero,
               pct_flagged_overlap, pct_reliable_overlap, artifact_label))

t3_cluster_order <- zero_summary %>%
  arrange(pct_CI_overlaps_zero) %>%
  pull(TC_cluster)

p_zero_overlap <- zero_summary %>%
  mutate(TC_cluster = factor(TC_cluster, levels = t3_cluster_order)) %>%
  pivot_longer(
    cols      = c(pct_flagged_overlap, pct_reliable_overlap),
    names_to  = "unit_type",
    values_to = "pct_overlap"
  ) %>%
  mutate(unit_type = recode(unit_type,
                            pct_flagged_overlap  = "Flagged units",
                            pct_reliable_overlap = "Reliable units")) %>%
  ggplot(aes(y = TC_cluster, x = pct_overlap, fill = unit_type)) +
  geom_col(position = "dodge", width = 0.65, alpha = 0.88) +
  geom_vline(xintercept = 50, linetype = "dashed",
             colour = "grey50", linewidth = 0.5) +
  annotate("text", x = 50, y = Inf,
           label = "50% overlap", hjust = -0.1, vjust = 1.5,
           size = 3, colour = "grey45") +
  scale_fill_manual(
    values = c("Flagged units" = "#E24B4A", "Reliable units" = "#185FA5"),
    name   = NULL
  ) +
  scale_x_continuous(limits = c(0, 105),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Test 3 — 95% CI overlap with zero per cluster",
    subtitle = paste0(
      "Among units passing detection threshold (≥", MIN_DETECT_CELLS,
      " cells AND ≥", MIN_DETECT_PROP * 100, "%)\n",
      ">50% overlap in flagged units = strong artifact indicator"
    ),
    x = "% of detections where 95% CI includes zero",
    y = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 8.5, colour = "grey35"),
    axis.text       = element_text(colour = "black"),
    legend.position = "top",
    legend.key.size = unit(0.4, "cm")
  )

print(p_zero_overlap)

# ══════════════════════════════════════════════════════════════════════════════
# 13. Combined artifact verdict — all three tests
# ══════════════════════════════════════════════════════════════════════════════

verdict <- cluster_rank %>%
  left_join(artifact_cor %>% select(TC_cluster, spearman_r, artifact_label,
                                    enrichment_in_flagged),
            by = "TC_cluster") %>%
  left_join(zero_summary %>% select(TC_cluster, pct_CI_overlaps_zero,
                                    pct_flagged_overlap),
            by = "TC_cluster") %>%
  # ── fix: clusters with no detections in flagged units get NA → treat as 0
  mutate(
    pct_flagged_overlap   = replace_na(pct_flagged_overlap,   0),
    pct_CI_overlaps_zero  = replace_na(pct_CI_overlaps_zero,  0),
    # ── fix: clusters absent from zero_summary entirely (0 detections anywhere)
    # get NA from the join — also treat as 0 (no evidence of anything)
    n_red_flags = (instability_rank <= ceiling(length(tc_cols) / 2)) +
      (artifact_label %in% c("likely artifact",
                             "possible artifact")) +
      (pct_flagged_overlap > 50),
    final_verdict = case_when(
      n_red_flags == 3 ~ "artifact (3/3 tests)",
      n_red_flags == 2 ~ "probable artifact (2/3 tests)",
      n_red_flags == 1 ~ "uncertain (1/3 tests)",
      TRUE             ~ "biology (0/3 tests)"
    ) %>% factor(levels = c("artifact (3/3 tests)",
                            "probable artifact (2/3 tests)",
                            "uncertain (1/3 tests)",
                            "biology (0/3 tests)"))
  ) %>%
  arrange(desc(n_red_flags))

message("\nFinal artifact verdict per cluster (", nrow(verdict), " clusters):")
print(verdict %>%
        select(TC_cluster, instability_rank, spearman_r,
               pct_CI_overlaps_zero, pct_flagged_overlap,
               n_red_flags, final_verdict))

verdict_colours <- c(
  "artifact (3/3 tests)"          = "#E24B4A",
  "probable artifact (2/3 tests)" = "#EF9F27",
  "uncertain (1/3 tests)"         = "#888780",
  "biology (0/3 tests)"           = "#185FA5"
)

t4_cluster_order <- verdict %>% arrange(n_red_flags) %>% pull(TC_cluster)

p_verdict <- verdict %>%
  mutate(TC_cluster = factor(TC_cluster, levels = t4_cluster_order)) %>%
  ggplot(aes(y = TC_cluster, x = n_red_flags, fill = final_verdict)) +
  geom_col(width = 0.65, alpha = 0.9) +
  geom_text(aes(x     = n_red_flags + 0.05,
                label = paste0(final_verdict,
                               " | r=", round(spearman_r, 2),
                               " | flagged=", round(pct_flagged_overlap, 0), "%",
                               " | CI∩0=", round(pct_CI_overlaps_zero, 0), "%")),
            hjust = 0, size = 2.6, colour = "grey25") +
  scale_fill_manual(values = verdict_colours, name = NULL, guide = "none") +
  scale_x_continuous(breaks = 0:3, limits = c(0, 8),
                     labels = c("0", "1", "2", "3")) +
  labs(
    title    = "Combined artifact verdict — all three tests",
    subtitle = paste0(
      "Detection threshold: ≥", MIN_DETECT_CELLS,
      " cells AND ≥", MIN_DETECT_PROP * 100, "% proportion\n",
      "Red flags: T1 = top-half instability | T2 = enriched in low-n | T3 = CI overlaps zero >50%\n",
      "Label shows: Spearman r | % detected in flagged units | % CI overlapping zero"
    ),
    x = "Number of red flags (out of 3)",
    y = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title    = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle = element_text(size = 8,  colour = "grey35"),
    axis.text     = element_text(colour = "black")
  )

print(p_verdict)

#### REDO WITH LOWER TRESHOLDS

# ── Rebuild T2 detection with a RELAXED threshold ─────────────────────────────
# The strict threshold (≥3 cells AND ≥1%) is correct for T3 (CI overlap)
# but kills T2 because flagged units (n<30) never pass it
# For T2 we only need ≥1 cell to ask "is this cluster present at all?"

MIN_DETECT_CELLS_T2 <- 1     # T2 only — just needs to be observed
MIN_DETECT_PROP_T2  <- 0.01  # T2 only — still require ≥1% proportion

detection_df_t2 <- observed_props_agg %>%
  left_join(tc_counts_per_cluster,
            by = c("unit_id", "TC_cluster")) %>%
  replace_na(list(n_cluster_cells = 0)) %>%
  left_join(instability_agg %>% select(unit_id, n_TC_cells, flagged,
                                       reliability),
            by = "unit_id") %>%
  mutate(
    detected = n_cluster_cells >= MIN_DETECT_CELLS_T2 &
      observed_prop   >= MIN_DETECT_PROP_T2,
    n_bin    = cut(n_TC_cells,
                   breaks         = c(0, 10, 30, 100, 500, Inf),
                   labels         = c("<10", "10–30", "30–100", "100–500", ">500"),
                   right          = TRUE,
                   include.lowest = TRUE)
  )

# Rebuild detection_rate and artifact_cor from the relaxed threshold
detection_rate <- detection_df_t2 %>%
  group_by(TC_cluster, n_bin, .drop = FALSE) %>%
  summarise(
    n_units        = n(),
    n_detected     = sum(detected),
    detection_rate = n_detected / n_units,
    .groups = "drop"
  )

artifact_cor <- detection_df_t2 %>%
  group_by(TC_cluster) %>%
  summarise(
    spearman_r   = cor(n_TC_cells, as.numeric(detected),
                       method = "spearman", use = "complete.obs"),
    p_value      = tryCatch(
      cor.test(n_TC_cells, as.numeric(detected),
               method = "spearman", exact = FALSE)$p.value,
      error = function(e) NA_real_
    ),
    pct_detected_flagged       = mean(detected[flagged == TRUE],  na.rm = TRUE) * 100,
    pct_detected_reliable      = mean(detected[flagged == FALSE], na.rm = TRUE) * 100,
    enrichment_in_flagged      = pct_detected_flagged - pct_detected_reliable,
    median_cells_when_detected = median(n_cluster_cells[detected], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    p_adj         = p.adjust(p_value, method = "BH"),
    artifact_flag = spearman_r < 0 & p_adj < 0.05,
    artifact_label = case_when(
      spearman_r < -0.3 & p_adj < 0.05  ~ "likely artifact",
      spearman_r < 0    & p_adj < 0.05  ~ "possible artifact",
      spearman_r < 0    & p_adj >= 0.05 ~ "inconclusive",
      TRUE                               ~ "likely biology"
    ) %>% factor(levels = c("likely artifact", "possible artifact",
                            "inconclusive",    "likely biology"))
  ) %>%
  arrange(spearman_r)

message("\nDetection rate artifact scores — T2 (relaxed threshold, most artifact-like first):")
print(artifact_cor %>%
        select(TC_cluster, spearman_r, p_adj, pct_detected_flagged,
               pct_detected_reliable, enrichment_in_flagged, artifact_label))

# ── T3 zero overlap uses the STRICT threshold (unchanged) ─────────────────────
# But NaN (= no detections in flagged units) stays NA, not replaced with 0
zero_overlap <- boot_results_agg %>%
  select(-n_TC_cells) %>%
  left_join(observed_props_agg %>% select(unit_id, TC_cluster, observed_prop),
            by = c("unit_id", "TC_cluster")) %>%
  left_join(tc_counts_per_cluster,
            by = c("unit_id", "TC_cluster")) %>%
  replace_na(list(observed_prop = 0, n_cluster_cells = 0)) %>%
  left_join(instability_agg %>% select(unit_id, n_TC_cells, flagged,
                                       reliability),
            by = "unit_id") %>%
  mutate(
    CI_overlaps_zero = boot_q025 <= 0,
    detected         = n_cluster_cells >= MIN_DETECT_CELLS &  # strict threshold
      observed_prop   >= MIN_DETECT_PROP
  )

zero_summary <- zero_overlap %>%
  filter(detected) %>%
  group_by(TC_cluster) %>%
  summarise(
    n_detected             = n(),
    n_CI_overlaps_zero     = sum(CI_overlaps_zero),
    pct_CI_overlaps_zero   = mean(CI_overlaps_zero) * 100,
    n_flagged_detected     = sum(flagged),
    n_flagged_overlap_zero = sum(CI_overlaps_zero & flagged),
    # Keep NaN as NA — "no detections in flagged units" ≠ "CI never overlaps zero"
    pct_flagged_overlap    = ifelse(sum(flagged) > 0,
                                    mean(CI_overlaps_zero[flagged]) * 100,
                                    NA_real_),
    pct_reliable_overlap   = ifelse(sum(!flagged) > 0,
                                    mean(CI_overlaps_zero[!flagged]) * 100,
                                    NA_real_),
    .groups = "drop"
  ) %>%
  left_join(artifact_cor %>% select(TC_cluster, artifact_label, spearman_r),
            by = "TC_cluster") %>%
  arrange(desc(pct_CI_overlaps_zero))

# ── Rebuild verdict ───────────────────────────────────────────────────────────
verdict <- cluster_rank %>%
  left_join(artifact_cor %>% select(TC_cluster, spearman_r, artifact_label,
                                    enrichment_in_flagged,
                                    pct_detected_flagged,
                                    pct_detected_reliable),
            by = "TC_cluster") %>%
  left_join(zero_summary %>% select(TC_cluster, pct_CI_overlaps_zero,
                                    pct_flagged_overlap),
            by = "TC_cluster") %>%
  mutate(
    # T3: NA (insufficient data in flagged units) → does not fire
    t1_flag = instability_rank <= ceiling(length(tc_cols) / 2),
    t2_flag = artifact_label %in% c("likely artifact", "possible artifact"),
    t3_flag = !is.na(pct_flagged_overlap) & pct_flagged_overlap > 50,
    n_red_flags = t1_flag + t2_flag + t3_flag,
    final_verdict = case_when(
      n_red_flags == 3 ~ "artifact (3/3 tests)",
      n_red_flags == 2 ~ "probable artifact (2/3 tests)",
      n_red_flags == 1 ~ "uncertain (1/3 tests)",
      TRUE             ~ "biology (0/3 tests)"
    ) %>% factor(levels = c("artifact (3/3 tests)",
                            "probable artifact (2/3 tests)",
                            "uncertain (1/3 tests)",
                            "biology (0/3 tests)"))
  ) %>%
  arrange(desc(n_red_flags))

message("\nFinal artifact verdict (", nrow(verdict), " clusters):")
print(verdict %>%
        select(TC_cluster, instability_rank, t1_flag, t2_flag, t3_flag,
               spearman_r, enrichment_in_flagged,
               pct_CI_overlaps_zero, n_red_flags, final_verdict))

verdict_colours <- c(
  "artifact (3/3 tests)"          = "#E24B4A",
  "probable artifact (2/3 tests)" = "#EF9F27",
  "uncertain (1/3 tests)"         = "#888780",
  "biology (0/3 tests)"           = "#185FA5"
)

t4_cluster_order <- verdict %>% arrange(n_red_flags) %>% pull(TC_cluster)

p_verdict <- verdict %>%
  mutate(TC_cluster = factor(TC_cluster, levels = t4_cluster_order)) %>%
  ggplot(aes(y = TC_cluster, x = n_red_flags, fill = final_verdict)) +
  geom_col(width = 0.65, alpha = 0.9) +
  geom_text(aes(
    x     = n_red_flags + 0.05,
    label = paste0(
      final_verdict,
      " | T1:", ifelse(t1_flag, "✓", "✗"),
      " T2:", ifelse(t2_flag, "✓", "✗"),
      " T3:", ifelse(t3_flag, "✓", ifelse(is.na(pct_flagged_overlap),
                                          "NA", "✗")),
      " | r=", round(spearman_r, 2),
      " | Δdetect=", round(enrichment_in_flagged, 0), "pp"
    )
  ),
  hjust = 0, size = 2.6, colour = "grey25") +
  scale_fill_manual(values = verdict_colours, name = NULL, guide = "none") +
  scale_x_continuous(breaks = 0:3, limits = c(0, 9),
                     labels = c("0", "1", "2", "3")) +
  labs(
    title    = "Combined artifact verdict — all three tests",
    subtitle = paste0(
      "T1: top-half instability rank | ",
      "T2: detection enriched in low-n (≥1 cell, ≥1%) | ",
      "T3: CI overlaps zero >50% in flagged units (≥3 cells, ≥1%)\n",
      "NA in T3 = cluster never detected in flagged units under strict threshold"
    ),
    x = "Number of red flags (out of 3)",
    y = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title    = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle = element_text(size = 7.5, colour = "grey35"),
    axis.text     = element_text(colour = "black")
  )

print(p_verdict)


##### REVISED

# Revised verdict that accounts for the rare-cluster T1/T3 blind spot
# Add a 4th criterion: strong T2 signal alone is sufficient for rare clusters

verdict_revised <- verdict %>%
  mutate(
    # Flag clusters that are strongly enriched in low-n even if T1/T3 can't fire
    strong_t2 = spearman_r < -0.3 & enrichment_in_flagged > 30,
    
    n_red_flags_revised = t1_flag + t2_flag + t3_flag + strong_t2,
    
    final_verdict_revised = case_when(
      n_red_flags_revised >= 3 ~ "artifact (≥3 criteria)",
      n_red_flags_revised == 2 ~ "probable artifact (2 criteria)",
      n_red_flags_revised == 1 ~ "uncertain (1 criterion)",
      TRUE                     ~ "biology (0 criteria)"
    ) %>% factor(levels = c("artifact (≥3 criteria)",
                            "probable artifact (2 criteria)",
                            "uncertain (1 criterion)",
                            "biology (0 criteria)"))
  ) %>%
  arrange(desc(n_red_flags_revised))

message("\nRevised verdict with rare-cluster correction:")
print(verdict_revised %>%
        select(TC_cluster, t1_flag, t2_flag, t3_flag, strong_t2,
               spearman_r, enrichment_in_flagged,
               n_red_flags_revised, final_verdict_revised))

verdict_colours_rev <- c(
  "artifact (≥3 criteria)"         = "#E24B4A",
  "probable artifact (2 criteria)" = "#EF9F27",
  "uncertain (1 criterion)"        = "#888780",
  "biology (0 criteria)"           = "#185FA5"
)

t4_order_rev <- verdict_revised %>% arrange(n_red_flags_revised) %>% pull(TC_cluster)

p_verdict_revised <- verdict_revised %>%
  mutate(TC_cluster = factor(TC_cluster, levels = t4_order_rev)) %>%
  ggplot(aes(y = TC_cluster, x = n_red_flags_revised, fill = final_verdict_revised)) +
  geom_col(width = 0.65, alpha = 0.9) +
  geom_text(aes(
    x     = n_red_flags_revised + 0.05,
    label = paste0(
      final_verdict_revised,
      " | T1:", ifelse(t1_flag,   "✓", "✗"),
      " T2:", ifelse(t2_flag,   "✓", "✗"),
      " T3:", ifelse(t3_flag,   "✓",
                     ifelse(is.na(pct_flagged_overlap), "NA", "✗")),
      " T2*:", ifelse(strong_t2, "✓", "✗"),
      " | r=", round(spearman_r, 2),
      " | Δ=", round(enrichment_in_flagged, 0), "pp"
    )
  ),
  hjust = 0, size = 2.5, colour = "grey25") +
  scale_fill_manual(values = verdict_colours_rev, name = NULL, guide = "none") +
  scale_x_continuous(breaks = 0:4, limits = c(0, 10.5)) +
  labs(
    title    = "Combined artifact verdict — revised (4 criteria)",
    subtitle = paste0(
      "T1: top-half instability | T2: detection enriched in low-n (≥1 cell, ≥1%) | ",
      "T3: CI overlaps zero >50% in flagged units\n",
      "T2*: strong rare-cluster artifact signal (r < −0.3 AND Δdetect > 30pp) — ",
      "corrects T1/T3 blind spot for very rare clusters"
    ),
    x = "Number of criteria met (out of 4)",
    y = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title    = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle = element_text(size = 7.5, colour = "grey35"),
    axis.text     = element_text(colour = "black")
  )

print(p_verdict_revised)


# ══════════════════════════════════════════════════════════════════════════════
# 14. Export
# ══════════════════════════════════════════════════════════════════════════════

instability_agg %>%
  filter(is_BM_DE) %>%
  arrange(n_TC_cells) %>%
  select(unit_id, Sample, Tissue, Timepoint, n_ROIs, n_TC_cells,
         mean_boot_SD, mean_CI_width, reliability, flagged, flag_reason) %>%
  write_csv("BM_DE_aggregated_TC_reliability.csv")

verdict %>%
  left_join(zero_summary %>% select(TC_cluster, n_detected,
                                    pct_CI_overlaps_zero,
                                    pct_flagged_overlap,
                                    pct_reliable_overlap),
            by = "TC_cluster") %>%
  write_csv("TC_cluster_artifact_verdict.csv")

# ── Save plots ─────────────────────────────────────────────────────────────────
n_tc  <- length(tc_cols)
n_bmd <- nrow(bm_de_units)

out <- "/home/rstudio/mnt_out/MapMet/figures/"

ggsave(paste0(out,"01_TC_instability_highlighted.pdf"),  
       p_highlighted,    width = 9,   height = 6.5, useDingbats = FALSE)
ggsave(paste0(out,"02_TC_BM_DE_ribbons.pdf"),            
       p_bm_de_ribbon,   width = 8,   height = 3 * ceiling(n_tc / 2), useDingbats = FALSE)
ggsave(paste0(out,"03_TC_BM_DE_CI_bars.pdf"),            
       p_table,          width = 8,   height = 0.35 * n_bmd + 2, useDingbats = FALSE)
ggsave(paste0(out,"04_T1_instability_stacked.pdf"),      
       p_cluster_sd,     width = 7,   height = 5, useDingbats = FALSE)
ggsave(paste0(out,"05_T1b_instability_violin.pdf"),      
p_cluster_violin, width = 7,   height = 5, useDingbats = FALSE)
ggsave(paste0(out,"06_T2a_detection_heatmap.pdf"),       
p_detection_heat, width = 7,   height = 4, useDingbats = FALSE)
ggsave(paste0(out,"07_T2b_detection_enrichment.pdf"),    
p_enrichment,     width = 7,   height = 4, useDingbats = FALSE)
ggsave(paste0(out,"08_T3_zero_overlap.pdf"),             
p_zero_overlap,   width = 7,   height = 4, useDingbats = FALSE)
ggsave(paste0(out,"09_T4_artifact_verdict.pdf"),         
       p_verdict_revised,        width = 8,   height = 4, useDingbats = FALSE)


message("THE END")



message("Now lets look at PT samples")


# ══════════════════════════════════════════════════════════════════════════════
# PT composition check for the 3 probable-artifact BM clusters
# Question: are CHGAhi TC, GD2lo TC, GATA3hi TC stably detected in PT?
# ══════════════════════════════════════════════════════════════════════════════

PT_LABEL          <- "PT"    
ARTIFACT_CLUSTERS <- tc_cols

# ── 1. Filter instability table to PT units ───────────────────────────────────
instability_pt <- instability_agg %>%
  filter(Tissue == PT_LABEL)

message("PT units found: ", nrow(instability_pt),
        " | reliability breakdown:")
print(count(instability_pt, reliability))

# ── 2. Detection summary for the 3 clusters in PT ────────────────────────────
# Uses the STRICT threshold (≥3 cells AND ≥1%) — same as T3
detection_pt <- observed_props_agg %>%
  filter(unit_id %in% instability_pt$unit_id) %>%
  left_join(tc_counts_per_cluster,
            by = c("unit_id", "TC_cluster")) %>%
  replace_na(list(n_cluster_cells = 0)) %>%
  left_join(instability_pt %>% select(unit_id, n_TC_cells, reliability,
                                      flagged),
            by = "unit_id") %>%
  mutate(
    detected = n_cluster_cells >= MIN_DETECT_CELLS &
      observed_prop   >= MIN_DETECT_PROP,
    n_bin    = cut(n_TC_cells,
                   breaks         = c(0, 10, 30, 100, 500, Inf),
                   labels         = c("<10", "10–30", "30–100", "100–500", ">500"),
                   right          = TRUE,
                   include.lowest = TRUE)
  )

# Detection rate per cluster per n-bin — PT only
detection_rate_pt <- detection_pt %>%
  filter(TC_cluster %in% ARTIFACT_CLUSTERS) %>%
  group_by(TC_cluster, n_bin, .drop = FALSE) %>%
  summarise(
    n_units        = n(),
    n_detected     = sum(detected),
    detection_rate = n_detected / n_units,
    .groups = "drop"
  )

# Spearman r for PT — compare direction vs BM
artifact_cor_pt <- detection_pt %>%
  filter(TC_cluster %in% ARTIFACT_CLUSTERS) %>%
  group_by(TC_cluster) %>%
  summarise(
    spearman_r_PT             = cor(n_TC_cells, as.numeric(detected),
                                    method = "spearman", use = "complete.obs"),
    p_value_PT                = tryCatch(
      cor.test(n_TC_cells, as.numeric(detected),
               method = "spearman", exact = FALSE)$p.value,
      error = function(e) NA_real_
    ),
    pct_detected_flagged_PT   = mean(detected[flagged == TRUE],  na.rm = TRUE) * 100,
    pct_detected_reliable_PT  = mean(detected[flagged == FALSE], na.rm = TRUE) * 100,
    mean_prop_when_detected   = mean(observed_prop[detected],    na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  mutate(p_adj_PT = p.adjust(p_value_PT, method = "BH"))

# Side-by-side BM vs PT Spearman r comparison
comparison_table <- artifact_cor_pt %>%
  left_join(
    artifact_cor %>%
      filter(TC_cluster %in% ARTIFACT_CLUSTERS) %>%
      select(TC_cluster, spearman_r_BM = spearman_r,
             pct_detected_flagged_BM   = pct_detected_flagged,
             pct_detected_reliable_BM  = pct_detected_reliable,
             enrichment_in_flagged_BM  = enrichment_in_flagged),
    by = "TC_cluster"
  ) %>%
  mutate(
    PT_signal = case_when(
      spearman_r_PT > 0.2  ~ "biology in PT (positive r)",
      spearman_r_PT > -0.1 ~ "neutral in PT",
      TRUE                 ~ "artifact in PT too"
    )
  )

message("\nBM vs PT comparison for probable-artifact clusters:")
print(comparison_table %>%
        select(TC_cluster, spearman_r_BM, spearman_r_PT, p_adj_PT,
               pct_detected_reliable_BM, pct_detected_reliable_PT,
               mean_prop_when_detected, PT_signal))

# ── 3. Plot A — detection heatmap PT only, 3 artifact clusters ───────────────
# Sort clusters by PT detection rate at >500 cells (most reliably detected first)
pt_heat_order <- detection_rate_pt %>%
  filter(n_bin == ">500") %>%
  arrange(desc(detection_rate)) %>%
  pull(TC_cluster)
# Handle any missing levels
pt_heat_order <- c(pt_heat_order,
                   setdiff(ARTIFACT_CLUSTERS, pt_heat_order))

p_pt_heat <- detection_rate_pt %>%
  mutate(TC_cluster = factor(TC_cluster, levels = pt_heat_order)) %>%
  ggplot(aes(x = n_bin, y = TC_cluster, fill = detection_rate)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(n_detected, "/", n_units)),
            size = 3.2, colour = "white") +
  scale_fill_gradientn(
    colours = c("#E6F1FB", "#378ADD", "#042C53"),
    limits  = c(0, 1),
    labels  = scales::percent_format(accuracy = 1),
    name    = "Detection\nrate"
  ) +
  labs(
    title    = "PT detection rate — probable-artifact BM clusters",
    subtitle = paste0("Detected = ≥", MIN_DETECT_CELLS,
                      " cells AND ≥", MIN_DETECT_PROP * 100,
                      "% proportion | if consistently high in PT = real biology"),
    x        = "TC cells in PT unit (pooled across ROIs)",
    y        = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title        = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle     = element_text(size = 8.5, colour = "grey35"),
    axis.text         = element_text(colour = "black"),
    legend.key.height = unit(0.8, "cm")
  )

# ── 4. Plot B — BM vs PT detection rate comparison (lollipop) ────────────────
compare_long <- comparison_table %>%
  select(TC_cluster,
         `BM — reliable` = pct_detected_reliable_BM,
         `PT — reliable` = pct_detected_reliable_PT) %>%
  pivot_longer(-TC_cluster, names_to = "context", values_to = "pct_detected")

p_compare_lollipop <- compare_long %>%
  mutate(TC_cluster = factor(TC_cluster, levels = rev(ARTIFACT_CLUSTERS))) %>%
  ggplot(aes(y = TC_cluster, x = pct_detected, colour = context)) +
  geom_line(aes(group = TC_cluster), colour = "grey70", linewidth = 1) +
  geom_point(size = 4.5, alpha = 0.9) +
  geom_text(aes(label = paste0(round(pct_detected, 0), "%")),
            vjust = -0.9, size = 3, colour = "grey25") +
  scale_colour_manual(
    values = c("BM — reliable" = "#E24B4A",
               "PT — reliable" = "#185FA5"),
    name   = NULL
  ) +
  scale_x_continuous(limits = c(0, 110),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Detection rate in reliable units: BM vs PT",
    subtitle = "Reliable = ≥30 TC cells pooled | large BM→PT gap = biology in PT, artifact in BM",
    x        = "% of reliable units where cluster is detected",
    y        = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 8.5, colour = "grey35"),
    axis.text       = element_text(colour = "black"),
    legend.position = "top",
    legend.key.size = unit(0.4, "cm")
  )

# ── 5. Plot C — bootstrap ribbon for 3 clusters, PT reliable units only ───────
# Shows actual proportion stability in PT — wide CI = rare even in PT
pt_reliable_ids <- instability_pt %>%
  filter(!flagged) %>%
  arrange(n_TC_cells) %>%
  pull(unit_id)

plot_df_pt_artifact <- boot_results_agg %>%
  filter(unit_id %in% pt_reliable_ids,
         TC_cluster %in% ARTIFACT_CLUSTERS) %>%
  select(-n_TC_cells) %>%
  left_join(observed_props_agg %>% select(unit_id, TC_cluster, observed_prop),
            by = c("unit_id", "TC_cluster")) %>%
  left_join(instability_pt %>% select(unit_id, reliability, n_TC_cells,
                                      n_ROIs),
            by = "unit_id") %>%
  replace_na(list(observed_prop = 0)) %>%
  mutate(
    unit_label = paste0(unit_id, "\n(n=", n_TC_cells, ")"),
    unit_label = factor(unit_label,
                        levels = unique(unit_label[order(
                          match(unit_id, pt_reliable_ids))]))
  )

p_pt_ribbon <- ggplot(plot_df_pt_artifact,
                      aes(x = unit_label, colour = reliability,
                          fill = reliability)) +
  geom_linerange(aes(ymin = boot_q025, ymax = boot_q975),
                 linewidth = 3, alpha = 0.18) +
  geom_linerange(aes(ymin = boot_q25,  ymax = boot_q75),
                 linewidth = 3, alpha = 0.45) +
  geom_point(aes(y = observed_prop), size = 2, alpha = 0.95) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3) +
  facet_wrap(~ TC_cluster, scales = "free_y", ncol = 1) +
  scale_colour_manual(values = rel_colours, name = "Reliability") +
  scale_fill_manual(values   = rel_colours, name = "Reliability") +
  scale_y_continuous(labels  = scales::percent_format(accuracy = 1)) +
  labs(
    title    = "Bootstrap stability in PT reliable units — probable-artifact clusters",
    subtitle = "Point = observed; thick = IQR; thin = 95% CI | sorted low → high TC cells\nNarrow CI + consistent point = biologically real in PT",
    x        = NULL,
    y        = "TC cluster proportion"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title       = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle    = element_text(size = 8,  colour = "grey35"),
    axis.text.x      = element_text(angle = 45, hjust = 1, size = 6.5,
                                    colour = "black"),
    strip.text       = element_text(face = "bold", size = 10),
    strip.background = element_blank(),
    legend.position  = "top",
    legend.key.size  = unit(0.4, "cm")
  )

print(p_pt_heat)
print(p_compare_lollipop)
print(p_pt_ribbon)

# ── 6. Export ─────────────────────────────────────────────────────────────────
write_csv(comparison_table, paste0(out,"artifact_clusters_BM_vs_PT_detection.csv"))

ggsave(paste0(out,"10_PT_detection_heatmap_artifact_clusters.pdf"), 
p_pt_heat,      width = 7, height = 3, useDingbats = FALSE)
ggsave(paste0(out,"11_BM_vs_PT_detection_lollipop.pdf"),           
p_compare_lollipop, width = 6, height = 4, useDingbats = FALSE)
ggsave(paste0(out,"12_PT_ribbon_artifact_clusters.pdf"),           
p_pt_ribbon, width = 10, height = 8, useDingbats = FALSE)

