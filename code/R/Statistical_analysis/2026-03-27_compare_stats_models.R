# comparing different statistical models
# accounting for composition

library(lme4)
library(lmerTest)
library(emmeans)
library(tidyverse)

# ══════════════════════════════════════════════════════════════════════════════
# 0. Parameters
# ══════════════════════════════════════════════════════════════════════════════

CLR_PSEUDOCOUNT <- 0.5   # added as 0.5/n_TC_cells per unit (Haldane-Anscombe)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Build modelling dataset — all three representations
# ══════════════════════════════════════════════════════════════════════════════

# Base: DE timepoint, reliable units only
base_df <- observed_props_agg %>%
  left_join(tc_counts_per_cluster,
            by = c("unit_id", "TC_cluster")) %>%
  replace_na(list(n_cluster_cells = 0)) %>%
  left_join(instability_agg %>% select(unit_id, Sample, Tissue, Timepoint,
                                       n_TC_cells, flagged, reliability),
            by = "unit_id") %>%
  filter(Timepoint == DE_LABEL, !flagged) %>%
  mutate(
    Tissue     = factor(Tissue, levels = c(PT_LABEL, BM_LABEL)),
    log_weight = log(n_TC_cells)
  )

message("Units per tissue in model:")
print(count(base_df %>% distinct(unit_id, Tissue), Tissue))

# ── Logit-transformed proportion ──────────────────────────────────────────────
logit_df <- base_df %>%
  mutate(
    prop_adj   = ifelse(observed_prop == 0,
                        CLR_PSEUDOCOUNT / n_TC_cells,
                        observed_prop),
    logit_prop = log(prop_adj / (1 - prop_adj))
  )

# ── CLR transform (manual — no external package needed) ───────────────────────
# clr(p_i) = log(p_i) - mean(log(p_j)) across all clusters per unit
# Removes the sum-to-1 compositional constraint

clr_df <- base_df %>%
  group_by(unit_id) %>%
  mutate(
    prop_adj  = ifelse(observed_prop == 0,
                       CLR_PSEUDOCOUNT / n_TC_cells,
                       observed_prop),
    log_prop  = log(prop_adj),
    clr_value = log_prop - mean(log_prop)   # deviation from geometric mean
  ) %>%
  ungroup()

# Sanity check: CLR values must sum to ~0 per unit
clr_check <- clr_df %>%
  group_by(unit_id) %>%
  summarise(clr_sum = sum(clr_value), .groups = "drop")
message("CLR sanity check — max absolute deviation from zero per unit: ",
        round(max(abs(clr_check$clr_sum)), 12),
        " (should be <1e-10)")

# ── Absolute count (log1p-transformed) ────────────────────────────────────────
# WARNING: BM vs PT comparison confounded by cytospin vs tissue section prep
# Valid for within-tissue comparisons only — included for completeness

count_df <- base_df %>%
  mutate(log_count = log1p(n_cluster_cells))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Generic LMM fitter — reused for all three representations
# ══════════════════════════════════════════════════════════════════════════════

fit_lmm_generic <- function(cluster, df, response_col,
                            weight_col = "log_weight",
                            model_name = "model") {
  
  d <- df %>%
    filter(TC_cluster == cluster) %>%
    rename(y = all_of(response_col),
           w = all_of(weight_col))
  
  n_bm <- sum(d$Tissue == BM_LABEL)
  n_pt <- sum(d$Tissue == PT_LABEL)
  
  if (n_bm < 3 || n_pt < 3 || var(d$y, na.rm = TRUE) == 0) {
    return(tibble(
      TC_cluster = cluster, model = model_name,
      estimate   = NA_real_, se = NA_real_,
      df_satt    = NA_real_, t_value = NA_real_,
      p_value    = NA_real_, mm_BM = NA_real_, mm_PT = NA_real_,
      n_BM       = n_bm, n_PT = n_pt,
      n_patients = n_distinct(d$Sample),
      n_paired   = NA_integer_,
      converged  = FALSE, note = "insufficient data"
    ))
  }
  
  n_paired <- d %>%
    distinct(Sample, Tissue) %>%
    count(Sample) %>%
    filter(n == 2) %>%
    nrow()
  
  fit <- tryCatch(
    suppressMessages(
      lmer(y ~ Tissue + (1 | Sample),
           data    = d,
           weights = w,
           REML    = TRUE,
           control = lmerControl(optimizer = "bobyqa",
                                 optCtrl   = list(maxfun = 1e5)))
    ),
    error = function(e) NULL
  )
  
  if (is.null(fit)) {
    return(tibble(
      TC_cluster = cluster, model = model_name,
      estimate   = NA_real_, se = NA_real_,
      df_satt    = NA_real_, t_value = NA_real_,
      p_value    = NA_real_, mm_BM = NA_real_, mm_PT = NA_real_,
      n_BM       = n_bm, n_PT = n_pt,
      n_patients = n_distinct(d$Sample), n_paired = n_paired,
      converged  = FALSE, note = "model failed to fit"
    ))
  }
  
  conv     <- length(fit@optinfo$conv$lme4) == 0
  coef_tb  <- as.data.frame(summary(fit)$coefficients)
  trow     <- paste0("Tissue", BM_LABEL)
  
  em <- tryCatch(
    as.data.frame(emmeans(fit, ~ Tissue, weights = "proportional")),
    error = function(e) NULL
  )
  
  mm_BM <- if (!is.null(em)) em$emmean[em$Tissue == BM_LABEL] else NA_real_
  mm_PT <- if (!is.null(em)) em$emmean[em$Tissue == PT_LABEL] else NA_real_
  
  tibble(
    TC_cluster = cluster,
    model      = model_name,
    estimate   = coef_tb[trow, "Estimate"],
    se         = coef_tb[trow, "Std. Error"],
    df_satt    = coef_tb[trow, "df"],
    t_value    = coef_tb[trow, "t value"],
    p_value    = coef_tb[trow, "Pr(>|t|)"],
    mm_BM      = mm_BM,
    mm_PT      = mm_PT,
    n_BM       = n_bm,
    n_PT       = n_pt,
    n_patients = n_distinct(d$Sample),
    n_paired   = n_paired,
    converged  = conv,
    note       = ifelse(conv, "OK", "convergence warning")
  )
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. Fit all three models for all clusters
# ══════════════════════════════════════════════════════════════════════════════

message("\nFitting Model 1: logit(proportion) ~ Tissue + (1|Patient)...")
results_logit <- map_dfr(
  tc_cols, fit_lmm_generic,
  df           = logit_df,
  response_col = "logit_prop",
  model_name   = "logit_proportion"
)

message("Fitting Model 2: CLR(proportion) ~ Tissue + (1|Patient)...")
results_clr <- map_dfr(
  tc_cols, fit_lmm_generic,
  df           = clr_df,
  response_col = "clr_value",
  model_name   = "CLR_proportion"
)

message("Fitting Model 3: log1p(count) ~ Tissue + (1|Patient) [cytospin confound]...")
results_count <- map_dfr(
  tc_cols, fit_lmm_generic,
  df           = count_df,
  response_col = "log_count",
  model_name   = "log_count"
)

# ── Apply BH correction within each model ─────────────────────────────────────
apply_BH <- function(df) {
  df %>%
    mutate(
      p_adj_BH  = p.adjust(p_value, method = "BH"),
      sig_label = case_when(
        p_adj_BH < 0.001 ~ "***",
        p_adj_BH < 0.01  ~ "**",
        p_adj_BH < 0.05  ~ "*",
        p_adj_BH < 0.10  ~ "†",
        TRUE             ~ "ns"
      ),
      direction = case_when(
        estimate > 0 ~ "BM > PT",
        estimate < 0 ~ "PT > BM",
        TRUE         ~ "equal"
      ),
      is_claim = TC_cluster %in% CLAIM_CLUSTERS
    ) %>%
    arrange(p_adj_BH)
}

results_logit <- apply_BH(results_logit)
results_clr   <- apply_BH(results_clr)
results_count <- apply_BH(results_count)

message("\n=== MODEL 1: Logit proportion ===")
print(results_logit %>%
        select(TC_cluster, estimate, se, p_value, p_adj_BH,
               sig_label, direction, mm_BM, mm_PT, converged))

message("\n=== MODEL 2: CLR proportion ===")
print(results_clr %>%
        select(TC_cluster, estimate, se, p_value, p_adj_BH,
               sig_label, direction, mm_BM, mm_PT, converged))

message("\n=== MODEL 3: log count [CYTOSPIN CONFOUND] ===")
print(results_count %>%
        select(TC_cluster, estimate, se, p_value, p_adj_BH,
               sig_label, direction, mm_BM, mm_PT, converged))

# Convergence warnings
all_fits <- bind_rows(results_logit, results_clr, results_count)
conv_warn <- all_fits %>% filter(!converged, !is.na(converged))
if (nrow(conv_warn) > 0) {
  message("\nConvergence warnings:")
  print(conv_warn %>% select(model, TC_cluster, note, n_BM, n_PT))
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. Comparison framework — flag disagreements across models
# ══════════════════════════════════════════════════════════════════════════════

comparison_all <- results_logit %>%
  select(TC_cluster, is_claim,
         logit_est  = estimate, logit_sig = sig_label,
         logit_dir  = direction, logit_padj = p_adj_BH) %>%
  left_join(
    results_clr %>%
      select(TC_cluster,
             clr_est  = estimate, clr_sig  = sig_label,
             clr_dir  = direction, clr_padj = p_adj_BH),
    by = "TC_cluster"
  ) %>%
  left_join(
    results_count %>%
      select(TC_cluster,
             count_est  = estimate, count_sig  = sig_label,
             count_dir  = direction, count_padj = p_adj_BH),
    by = "TC_cluster"
  ) %>%
  mutate(
    logit_clr_agree_dir = logit_dir == clr_dir,
    logit_clr_agree_sig = (logit_padj < 0.05) == (clr_padj < 0.05),
    count_agrees_logit  = count_dir == logit_dir,
    verdict = case_when(
      logit_clr_agree_dir & logit_clr_agree_sig & logit_padj < 0.05
      ~ "consistent — significant",
      logit_clr_agree_dir & logit_clr_agree_sig
      ~ "consistent — not significant",
      !logit_clr_agree_dir
      ~ "DIRECTION CONFLICT — logit vs CLR",
      logit_clr_agree_dir & !logit_clr_agree_sig
      ~ "direction agrees, significance differs",
      TRUE ~ "review manually"
    )
  )

message("\nModel comparison — full table:")
print(comparison_all %>%
        select(TC_cluster, logit_sig, clr_sig, count_sig,
               logit_clr_agree_dir, logit_clr_agree_sig,
               count_agrees_logit, verdict, is_claim))

message("\nDirection conflicts between logit and CLR:")
conflicts <- comparison_all %>% filter(!logit_clr_agree_dir)
if (nrow(conflicts) > 0) {
  print(conflicts %>%
          select(TC_cluster, logit_dir, logit_sig,
                 clr_dir, clr_sig, verdict))
} else {
  message("None — all clusters agree on direction between logit and CLR")
}

# ══════════════════════════════════════════════════════════════════════════════
# 5. Plot A — three-model forest comparison (z-scored for visual comparability)
# ══════════════════════════════════════════════════════════════════════════════

all_results <- bind_rows(results_logit, results_clr, results_count) %>%
  group_by(model) %>%
  mutate(
    sd_est     = sd(estimate, na.rm = TRUE),
    z_estimate = (estimate - mean(estimate, na.rm = TRUE)) / sd_est,
    z_se       = se / sd_est,
    model_label = recode(model,
                         logit_proportion = "Logit proportion",
                         CLR_proportion   = "CLR proportion",
                         log_count        = "log count\n[cytospin confound]") %>%
      factor(levels = c("Logit proportion",
                        "CLR proportion",
                        "log count\n[cytospin confound]"))
  ) %>%
  ungroup()

cluster_order_forest <- results_clr %>%
  arrange(estimate) %>%
  pull(TC_cluster)

model_colours <- c(
  "Logit proportion"               = "#185FA5",
  "CLR proportion"                 = "#1D9E75",
  "log count\n[cytospin confound]" = "#888780"
)

p_three_model <- all_results %>%
  mutate(TC_cluster = factor(TC_cluster, levels = cluster_order_forest)) %>%
  ggplot(aes(y = TC_cluster, x = z_estimate,
             colour = model_label, shape = model_label)) +
  geom_vline(xintercept = 0, colour = "grey50", linewidth = 0.5) +
  geom_errorbarh(aes(xmin = z_estimate - 1.96 * z_se,
                     xmax = z_estimate + 1.96 * z_se),
                 height   = 0.2, linewidth = 0.5, alpha = 0.6,
                 position = position_dodge(width = 0.6)) +
  geom_point(size     = 3, alpha = 0.9,
             position = position_dodge(width = 0.6)) +
  geom_text(aes(x     = z_estimate + 1.96 * z_se + 0.1,
                label = sig_label),
            hjust    = 0, size = 2.8,
            position = position_dodge(width = 0.6)) +
  scale_colour_manual(values = model_colours, name = "Model") +
  scale_shape_manual(values  = c(16, 17, 15),  name = "Model") +
  scale_x_continuous(expand  = expansion(mult = c(0.05, 0.2))) +
  labs(
    title    = "Three-model comparison: BM vs PT enrichment — all TC clusters",
    subtitle = paste0(
      "Estimates z-scored within model for visual comparability | ",
      "positive = BM enriched\n",
      "Logit + CLR: compositionally-corrected | ",
      "log count: confounded by cytospin vs section prep"
    ),
    x = "Standardised estimate (z-score within model, 95% CI)",
    y = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title      = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle   = element_text(size = 8,  colour = "grey35"),
    axis.text       = element_text(colour = "black"),
    legend.position = "right",
    legend.key.size = unit(0.4, "cm")
  )

# ══════════════════════════════════════════════════════════════════════════════
# 6. Plot B — model agreement heatmap
# ══════════════════════════════════════════════════════════════════════════════

sig_matrix <- all_results %>%
  mutate(
    cell_label = paste0(sig_label, "\n",
                        ifelse(direction == "BM > PT", "↑BM", "↓PT")),
    fill_val   = case_when(
      p_adj_BH < 0.05  & direction == "BM > PT" ~  1,
      p_adj_BH < 0.05  & direction == "PT > BM" ~ -1,
      p_adj_BH < 0.10  & direction == "BM > PT" ~  0.5,
      p_adj_BH < 0.10  & direction == "PT > BM" ~ -0.5,
      TRUE                                        ~  0
    ),
    TC_cluster = factor(TC_cluster, levels = cluster_order_forest)
  )

p_verdict_heat <- ggplot(sig_matrix,
                         aes(x = model_label, y = TC_cluster,
                             fill = fill_val)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = cell_label), size = 2.5, lineheight = 0.9,
            colour = "white") +
  # Red outline for claim clusters
  geom_rect(
    data = sig_matrix %>%
      filter(is_claim) %>%
      distinct(TC_cluster) %>%
      mutate(ynum = as.numeric(TC_cluster)),
    aes(ymin = ynum - 0.5, ymax = ynum + 0.5,
        xmin = 0.5, xmax = 3.5),
    fill = NA, colour = "#E24B4A",
    linewidth = 0.9, inherit.aes = FALSE
  ) +
  scale_fill_gradientn(
    colours = c("#042C53", "#378ADD", "#E6F1FB",
                "#FAECE7", "#D85A30", "#4A1B0C"),
    values  = scales::rescale(c(-1, -0.5, -0.05, 0.05, 0.5, 1)),
    limits  = c(-1, 1),
    name    = "Direction\n& strength",
    labels  = c("PT***", "", "ns", "", "BM***")
  ) +
  labs(
    title    = "Model agreement heatmap — BM vs PT, all three models",
    subtitle = "Red outline = claim clusters | ↑BM = BM enriched | ↓PT = PT enriched\nlog count model confounded by BM cytospin vs PT section preparation",
    x        = NULL,
    y        = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title        = element_text(size = 12, face = "bold", colour = "black"),
    plot.subtitle     = element_text(size = 8.5, colour = "grey35"),
    axis.text.x       = element_text(colour = "black", size = 9),
    axis.text.y       = element_text(colour = "black"),
    legend.key.height = unit(1.2, "cm")
  )

# ══════════════════════════════════════════════════════════════════════════════
# 7. Plot C — LMM marginal means for claim clusters (logit + CLR side by side)
# ══════════════════════════════════════════════════════════════════════════════

inv_logit <- function(x) 1 / (1 + exp(-x))

# Back-transform logit marginal means to proportion scale
claim_logit_mm <- results_logit %>%
  filter(is_claim) %>%
  select(TC_cluster, mm_BM, mm_PT, se, p_adj_BH, sig_label) %>%
  pivot_longer(cols = c(mm_BM, mm_PT),
               names_to = "Tissue", values_to = "emmean") %>%
  mutate(
    Tissue      = recode(Tissue, mm_BM = BM_LABEL, mm_PT = PT_LABEL),
    Tissue      = factor(Tissue, levels = c(PT_LABEL, BM_LABEL)),
    prop_pct    = inv_logit(emmean) * 100,
    ci_lo       = inv_logit(emmean - 1.96 * se) * 100,
    ci_hi       = inv_logit(emmean + 1.96 * se) * 100,
    model_label = "Logit proportion",
    TC_cluster  = factor(TC_cluster, levels = CLAIM_CLUSTERS)
  )

# CLR marginal means stay on CLR scale (no natural back-transform)
claim_clr_mm <- results_clr %>%
  filter(is_claim) %>%
  select(TC_cluster, mm_BM, mm_PT, se, p_adj_BH, sig_label) %>%
  pivot_longer(cols = c(mm_BM, mm_PT),
               names_to = "Tissue", values_to = "emmean") %>%
  mutate(
    Tissue      = recode(Tissue, mm_BM = BM_LABEL, mm_PT = PT_LABEL),
    Tissue      = factor(Tissue, levels = c(PT_LABEL, BM_LABEL)),
    prop_pct    = emmean,    # CLR scale — label axis accordingly
    ci_lo       = emmean - 1.96 * se,
    ci_hi       = emmean + 1.96 * se,
    model_label = "CLR proportion",
    TC_cluster  = factor(TC_cluster, levels = CLAIM_CLUSTERS)
  )

# Significance labels
sig_logit <- results_logit %>%
  filter(is_claim) %>%
  mutate(
    TC_cluster  = factor(TC_cluster, levels = CLAIM_CLUSTERS),
    model_label = "Logit proportion",
    y_pos       = inv_logit(pmax(mm_BM, mm_PT)) * 100 * 1.2,
    label       = paste0(sig_label, "\nΔ=",
                         round(inv_logit(mm_BM) * 100 -
                                 inv_logit(mm_PT) * 100, 1), "pp")
  )

sig_clr <- results_clr %>%
  filter(is_claim) %>%
  mutate(
    TC_cluster  = factor(TC_cluster, levels = CLAIM_CLUSTERS),
    model_label = "CLR proportion",
    y_pos       = pmax(mm_BM, mm_PT) * 1.3,
    label       = paste0(sig_label, "\nΔ=",
                         round(mm_BM - mm_PT, 2), " CLR units")
  )

plot_claim_mm <- function(df, sig_df, y_label, title_suffix) {
  ggplot(df, aes(x = Tissue, y = prop_pct,
                 colour = Tissue, fill = Tissue)) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                  width = 0.15, linewidth = 0.7, alpha = 0.8) +
    geom_point(size = 4.5, alpha = 0.95) +
    geom_line(aes(group = TC_cluster), colour = "grey60",
              linewidth = 0.5, linetype = "dashed") +
    geom_text(data    = sig_df,
              aes(x   = 1.5, y = y_pos, label = label),
              size    = 2.8, colour = "grey25",
              inherit.aes = FALSE) +
    facet_wrap(~ TC_cluster, scales = "free_y", ncol = 3) +
    scale_colour_manual(
      values = c("BM" = "#E24B4A", "PT" = "#185FA5"), name = NULL
    ) +
    scale_fill_manual(
      values = c("BM" = "#E24B4A", "PT" = "#185FA5"), name = NULL
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.3))) +
    labs(
      title    = paste0("LMM marginal means — claim clusters (", title_suffix, ")"),
      subtitle = paste0(
        "logit(prop) ~ Tissue + (1|Patient) | weights = log(n_TC_cells)\n",
        "Error bars = 95% CI | BH-corrected p-values"
      ),
      x = NULL, y = y_label
    ) +
    theme_classic(base_size = 11) +
    theme(
      plot.title       = element_text(size = 12, face = "bold", colour = "black"),
      plot.subtitle    = element_text(size = 8,  colour = "grey35"),
      strip.text       = element_text(face = "bold", size = 10),
      strip.background = element_blank(),
      axis.text        = element_text(colour = "black"),
      legend.position  = "none"
    )
}

p_claim_logit <- plot_claim_mm(
  claim_logit_mm, sig_logit,
  y_label      = "Estimated marginal mean proportion (%)",
  title_suffix = "logit scale, back-transformed"
)

p_claim_clr <- plot_claim_mm(
  claim_clr_mm, sig_clr,
  y_label      = "Estimated marginal mean (CLR units)",
  title_suffix = "CLR scale"
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. Print all plots
# ══════════════════════════════════════════════════════════════════════════════

print(p_three_model)
print(p_verdict_heat)
print(p_claim_logit)
print(p_claim_clr)

# ══════════════════════════════════════════════════════════════════════════════
# 9. Export
# ══════════════════════════════════════════════════════════════════════════════

bind_rows(results_logit, results_clr, results_count) %>%
  select(model, TC_cluster, estimate, se, df_satt, t_value,
         p_value, p_adj_BH, sig_label, direction,
         mm_BM, mm_PT, n_BM, n_PT, n_patients, n_paired,
         converged, note, is_claim) %>%
  write_csv("three_model_LMM_results.csv")

comparison_all %>%
  write_csv("three_model_agreement_table.csv")

# ggsave("19_three_model_forest.pdf",      p_three_model,  width = 10, height = 5,   useDingbats = FALSE)
# ggsave("20_model_agreement_heatmap.pdf", p_verdict_heat, width = 7,  height = 5,   useDingbats = FALSE)
# ggsave("21_claim_logit_marginal.pdf",    p_claim_logit,  width = 9,  height = 4.5, useDingbats = FALSE)
# ggsave("22_claim_clr_marginal.pdf",      p_claim_clr,    width = 9,  height = 4.5, useDingbats = FALSE)

# summary

final_three_model_verdict <- tribble(
  ~TC_cluster,           ~logit,    ~CLR,      ~conclusion,
  "CD24+ marker-lo TC",  "BM ** ",  "ns",      "COMPOSITIONAL ARTEFACT — enrichment disappears after CLR correction",
  "CD24- marker-lo TC",  "BM *  ",  "ns",      "COMPOSITIONAL ARTEFACT — enrichment disappears after CLR correction",
  "early SYM-like TC",   "PT ** ",  "PT ***",  "OPPOSITE DIRECTION — significantly depleted in BM, dominant in PT"
)

print(final_three_model_verdict)
write_csv(final_three_model_verdict, 
          "final_claim_verdict_three_model.csv")


### INTERPRETATION:

# I used the logit-transformed proportions in a linear mixed-effects model 
# with patient random intercepts and log(n_TC_cells) weighting, 
# CD24+ and CD24- marker-lo TC clusters appeared enriched in BM at diagnosis 
# (Δ=+11.4pp and +6.6pp respectively, BH p<0.01 and p<0.05). 
# However, after centered log-ratio transformation to account for the 
# compositional constraint of proportional data, neither cluster showed 
# significant BM enrichment (CLR Δ=+0.47 and +0.34 units, both ns), 
# indicating that their apparent proportional increase in BM reflects depletion 
# of other TC clusters rather than genuine clonal expansion of the marker-low population. 
# Early SYM-like TC showed consistent and significant PT enrichment across both models 
# (logit Δ=-10pp **, CLR Δ=-1.48 units ***), directly contradicting the proposed 
# preferential BM metastasis of this subtype. These findings do not support a 
# model of selective BM seeding by marker-lo or early SYM-like TC cells at diagnosis.


