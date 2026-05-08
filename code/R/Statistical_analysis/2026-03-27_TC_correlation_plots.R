
# 2026-03-27

metadata <- read.table("/home/rstudio/mnt_data/MapMet/lazic.imc.2024/metadata/20231128_metadata_complete.csv",sep =",",header = TRUE,row.names = 1)
imc <- readRDS("/home/rstudio/mnt_data/MapMet/lazic.imc.2024/Rds/imc_seurat.rds")
id_parts <- strsplit(as.character(imc$sample_id), "_")
unique(id_parts)
imc$Sample <- sapply(id_parts, function(x) x[3])
unique(imc$Sample)


# subset to DE samples only


# plot scatterplot of cells
library(tidyverse)
library(ggrepel)

# ── 0. Reproducibility ────────────────────────────────────────────────────────
set.seed(42)

# ── 1. Build per-sample proportion table ──────────────────────────────────────
ct_counts <- table(seu_meta$Sample, seu_meta$celltype)
ct_prop   <- prop.table(ct_counts, margin = 1)      # row-wise: fractions sum to 1 per sample
ct_df     <- as.data.frame.matrix(ct_prop)
ct_df$Sample <- rownames(ct_df)

# ── 2. Detect TC-matching column(s) ───────────────────────────────────────────
tc_pattern  <- "TC|[Tt]umor"                      
tc_cols     <- grep(tc_pattern, colnames(ct_df), value = TRUE)
message("TC-matching celltypes found:\n  ", paste(tc_cols, collapse = "\n  "))

# ── 3. Compute per-sample summary values ──────────────────────────────────────
# X axis: total fraction of all TC-matching cells per sample
# Y axis: fraction of the single specific TC cluster you called
#         → replace tc_cols[1] with your exact cluster name, e.g. "NB_TC" or "TC_adrenergic"
tc_specific <- tc_cols[1]   # <── EDIT THIS to your cluster of interest

plot_df <- ct_df %>%
  mutate(
    pct_TC_all     = rowSums(across(all_of(tc_cols))) * 100,   # X
    pct_TC_cluster = .data[[tc_specific]] * 100                # Y
  ) %>%
  select(Sample, pct_TC_all, pct_TC_cluster)

# ── 4. Correlation test ────────────────────────────────────────────────────────
# Use Spearman if you expect non-normality / outliers (common with n < 20 samples)
cor_test  <- cor.test(plot_df$pct_TC_all, plot_df$pct_TC_cluster,
                      method = "pearson")            # swap "spearman" if needed
r_val     <- round(cor_test$estimate, 3)
p_val     <- signif(cor_test$p.value, 3)
p_label   <- ifelse(cor_test$p.value < 0.001, "p < 0.001", paste0("p = ", p_val))
ann_label <- paste0("r = ", r_val, "\n", p_label)

# ── 5. Linear model for slope/intercept (optional: display in subtitle) ────────
lm_fit <- lm(pct_TC_cluster ~ pct_TC_all, data = plot_df)
slope  <- round(coef(lm_fit)[2], 3)
r2     <- round(summary(lm_fit)$r.squared, 3)

# ── 6. Plot ────────────────────────────────────────────────────────────────────
p <- ggplot(plot_df, aes(x = pct_TC_all, y = pct_TC_cluster)) +
  
  # CI ribbon first so points sit on top
  geom_smooth(method = "lm", se = TRUE,
              color    = "#185FA5",
              fill     = "#B5D4F4",
              linewidth = 0.8,
              alpha    = 0.25) +
  
  geom_point(size = 3, alpha = 0.85, color = "#378ADD") +
  
  # Sample labels — repel avoids overlap
  geom_text_repel(aes(label = Sample),
                  size       = 2.8,
                  max.overlaps = 20,
                  color      = "grey35",
                  segment.color = "grey70",
                  segment.size  = 0.3) +
  
  # Correlation annotation — bottom-right
  annotate("text",
           x = Inf, y = -Inf,
           hjust = 1.1, vjust = -0.8,
           label = ann_label,
           size  = 3.5, color = "grey25",
           fontface = "italic") +
  
  labs(
    title    = paste0("Tumor cell fraction vs. called cluster: ", tc_specific),
    subtitle = paste0("slope = ", slope, "  |  R² = ", r2),
    x        = "Total TC fraction per sample (%)",
    y        = paste0(tc_specific, " fraction per sample (%)")
  ) +
  
  theme_classic(base_size = 12) +
  theme(
    plot.title    = element_text(size = 12, face = "bold",  color = "black"),
    plot.subtitle = element_text(size = 10, color = "grey40"),
    axis.text     = element_text(color = "black"),
    axis.title    = element_text(color = "black"),
    axis.ticks    = element_line(color = "black", linewidth = 0.4),
    axis.line     = element_line(color = "black", linewidth = 0.4)
  )

print(p)


