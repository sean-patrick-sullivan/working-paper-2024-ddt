# -------------------------------------------------------------------------
# Appendix for Holt, Kwiatkowski & Sullivan (2024)
#
# The following R code produces Figure 1 and Appendix Figures A1-A3 of 
# Holt, Kwiatkowski, & Sullivan (2024). Input data in csv format is produced 
# by Monte Carlo simulations in Python.
#
# Scripted for R version 4.2.2
# -------------------------------------------------------------------------



# Preamble ----------------------------------------------------------------

# Load tidyverse library
library(tidyverse)

# Import data from Monte Carlo simulations
setwd("~/2024_ddt")
dta_in_baseline <- read_csv("baseline.csv", col_names = TRUE)
dta_in_groups_4 <- read_csv("groups_4.csv", col_names = TRUE)
dta_in_asymmetric_1 <- read_csv("asymmetric_1.csv", col_names = TRUE)
dta_in_asymmetric_2 <- read_csv("asymmetric_2.csv", col_names = TRUE)

# Map distribution labels to indexes
dist_levels <- seq(1,8)
dist_labels <- c(
  "Normal",
  "Uniform",
  "Normal with outliers",
  "Normal plus uniform",
  "Logistic",
  "Cauchy",
  "Gumbel",
  "Exponential"
)

# Map test labels to indexes
test_levels <- c("D", "J")
test_labels <- c("Directional Difference", "Jonckheere-Terpstra")

# Utility function to reshape raw data and apply labels
format_dta_in <- function (dta_in) {
  dta_out <- pivot_longer(
    dta_in,
    !d,
    names_to = c("distribution", "test"),
    names_sep = ' ',
    values_to = "probability"
  ) %>%
    mutate(distribution = factor(
      (as.integer(distribution)+1), 
      levels = dist_levels, 
      labels = dist_labels)
    ) %>%
    mutate(test = factor(
      test,
      levels = test_levels,
      labels = test_labels
    ))
  return( dta_out)
}

# Populate formatted data tibbles using format_dta_in()
dta_baseline <- format_dta_in(dta_in_baseline)
dta_groups_4 <- format_dta_in(dta_in_groups_4)
dta_asymmetric_1 <- format_dta_in(dta_in_asymmetric_1)
dta_asymmetric_2 <- format_dta_in(dta_in_asymmetric_2)


# Produce figures ---------------------------------------------------------

# Utility function to produce 4x2 matrix of power plots for supplied data
print_plots <- function (dta, selected_levels = dist_levels, n_cols = 2) {
  ggplot(dta %>% filter(as.integer(distribution) %in% selected_levels)) +
    theme_minimal() +
    theme(
      panel.border = element_rect(fill = NA, color = "gray70", linewidth = NULL),
      legend.position="bottom",
      legend.title=element_blank(),
      axis.title.x = element_text(margin = margin(t = 15)),
      axis.title.y = element_text(margin = margin(r = 10))
    ) +
    labs(
      x = "Location shift (size of actual treatment effect)",
      y = "Power (probability of rejecting null hypothesis)"
    ) +
    geom_line(aes(x = d, y = probability, color = test, linetype = test), 
              linewidth = 0.9) +
    scale_color_manual(values=c('black', 'gray60')) +
    facet_wrap(
      ~ distribution, 
      ncol = n_cols,
      labeller = labeller(distribution = dist_labels) )
}


# Produce Figure 1 using print_plots()
print_plots(dta_baseline)

# Produce Figures A1-A3 using print_plots()
print_plots(dta_groups_4)
print_plots(dta_asymmetric_1)
print_plots(dta_asymmetric_2)



# EOF