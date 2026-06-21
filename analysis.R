# Amazon Sales Signal Analysis: Best Seller Prediction at Scale
# Contributors: Dulce Ximena Cid Sanabria, Marco Alejandro Ortiz
# =============================================================================

# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------
library(tidyverse)
library(glmnet)
library(mgcv)
library(pROC)
library(PRROC)
library(Matrix)
library(scales)
library(gridExtra)
library(e1071)
library(lme4)
library(broom.mixed)

# -----------------------------------------------------------------------------
# 1. Data Loading & Cleaning
# -----------------------------------------------------------------------------
products   <- read_csv("amazon_products.csv", show_col_types = FALSE)
categories <- read_csv("amazon_categories.csv", show_col_types = FALSE)

products <- products %>%
  select(-imgUrl, -productURL, -listPrice) %>%
  mutate(isBestSeller = as.integer(isBestSeller %in% c("True", TRUE, 1))) %>%
  filter(price > 0, stars > 0) %>%
  left_join(categories, by = c("category_id" = "id")) %>%
  mutate(
    log_price     = log(price),
    log_reviews   = log1p(reviews),
    category_name = factor(category_name)
  )

# Coarse category grouping
classify_big <- function(x) {
  x <- as.character(x)
  dplyr::case_when(
    grepl("Smart Home", x)                                                                              ~ "Smart Home",
    grepl("Wearable|Virtual Reality", x)                                                                ~ "Wearable & VR",
    grepl("Nintendo|PlayStation|Xbox|Wii|PC Games|Mac Games|Video Game|Sony PSP|Online Video Game", x) ~ "Video Games",
    grepl("Computer|Phone|Camera|Audio|TV|Television|Headphone|Earbud|Tablet|Laptop|eBook|GPS|Office Electronics|Portable Audio|Video Project|Data Storage|Vehicle Electronics|Security & Surveillance|Kids' Electronics|Legacy Systems", x) ~ "Electronics",
    grepl("Luggage|Suitcase|Backpack|Travel|Messenger Bag|Garment Bag|Rain Umbrella", x)                ~ "Luggage & Travel",
    grepl("Clothing|Shoes|Jewelry|Watches|Handbag|Accessor|School Uniform|Sunglass", x)                 ~ "Apparel & Accessories",
    grepl("Beauty|Makeup|Hair Care|Skin Care|Personal Care|Perfume|Fragrance|Oral Care|Shaving|Foot, Hand|Nail Care", x) ~ "Beauty & Personal Care",
    grepl("Health|Medical|Vitamin|Diet|Sports Nutrition|Wellness|Vision Products|Sexual Wellness", x)   ~ "Health & Wellness",
    grepl("Toy|Doll|Puzzle|Stuffed Animal|Kids' Play|Building Toy|Learning|Finger Toy|Novelty Toy|Puppet|Slot Car|Tricycle|Games & Acc", x) ~ "Toys & Games",
    grepl("Kitchen|Bath Product|Bedding|Furniture|Décor|Decor|Lighting|Light Bulb|Vacuum|Ironing|Heating, Cooling|Home Appliance|Home Storage|Home Audio|Wall Art|Kids' Home|Kids' Furniture", x) ~ "Home & Kitchen",
    grepl("Cleaning|Household", x)                                                                      ~ "Household Supplies",
    grepl("Tool|Hardware|Building Supplies|Fastener|Cutting|Welding|Power Transmission|Hydraulics|Pneumatics|Plumbing|Pump|Paint, Wall", x) ~ "Tools & Home Improvement",
    grepl("Industrial|Lab|Scientific|Abrasive|Additive Manufacturing|Material Handling|Packaging|Retail Store|Occupational|Electrical Equipment|Electronic Components|Food Service|Janitorial|Filtration|Safety|Measuring|Test|Heavy Duty|Commercial Door|Oils|Professional Medical|Professional Dental|Science Educat", x) ~ "Industrial & Scientific",
    grepl("Automotive|Car Care|Motorcycle|RV |Vehicle", x)                                              ~ "Automotive",
    grepl("Sports|Outdoor|Fitness", x)                                                                  ~ "Sports & Outdoor",
    grepl("Pet|Dog|Cat |Bird|Fish|Reptile|Horse|Aquatic|Small Animal", x)                               ~ "Pet Supplies",
    grepl("Baby|Infant|Toddler|Child Safety|Nursery|Pregnancy|Toilet Training|Maternity", x)            ~ "Baby & Maternity",
    grepl("Arts|Crafts|Sewing|Knitting|Beading|Scrapbooking|Painting, Drawing|Fabric|Needlework|Printmaking", x) ~ "Arts & Crafts",
    grepl("Office|Stationery|Gift Wrapping|Party|Seasonal|Gift Card", x)                                ~ "Office & Party",
    TRUE                                                                                                ~ "Other"
  )
}

products <- products %>%
  mutate(big_category = factor(classify_big(category_name)))

# Summary
tibble(
  metric = c("Raw rows", "After price>0 & stars>0",
             "Best Seller prevalence", "Missing values (post-clean)"),
  value  = c(comma(nrow(products)), comma(nrow(products)),
             percent(mean(products$isBestSeller), accuracy = 0.01),
             comma(sum(is.na(products))))
) %>% as.data.frame()

# Descriptive statistics
products %>%
  summarise(
    across(c(price, stars, reviews),
           list(Min = min, Median = median, Mean = mean, Max = max, SD = sd),
           .names = "{.col}__{.fn}")
  ) %>%
  pivot_longer(everything(), names_to = c("Variable", "stat"), names_sep = "__") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(Variable = recode(Variable,
                           price   = "Price (USD)",
                           stars   = "Stars (1-5)",
                           reviews = "Review count")) %>%
  as.data.frame()

# -----------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# -----------------------------------------------------------------------------

# Class imbalance
products %>%
  count(isBestSeller) %>%
  mutate(label = ifelse(isBestSeller == 1, "Best Seller", "Other")) %>%
  ggplot(aes(label, n, fill = label)) +
  geom_col(width = 0.5) +
  scale_y_continuous(labels = comma) +
  labs(title = "Amazon Sales Signals: Class Imbalance",
       subtitle = "Only 0.67% of 1.26M listings carry Best Seller status",
       x = NULL, y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# Price and reviews: raw vs. log-transformed
products %>%
  select(price, reviews, log_price, log_reviews) %>%
  pivot_longer(everything(), names_to = "var", values_to = "value") %>%
  mutate(var = factor(var, levels = c("price", "log_price", "reviews", "log_reviews"))) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 50, fill = "steelblue") +
  facet_wrap(~ var, scales = "free", ncol = 2) +
  labs(title = "Amazon Sales Signals: Price & Review Distributions",
       subtitle = "Raw vs. log-transformed — both predictors are heavily right-skewed",
       x = NULL, y = NULL) +
  theme_minimal()

# log(price) and stars distributions
products %>%
  select(log_price, stars) %>%
  pivot_longer(everything(), names_to = "var", values_to = "value") %>%
  mutate(var = recode(var, log_price = "log(price)", stars = "Stars")) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 40, fill = "steelblue", color = "grey30") +
  facet_wrap(~ var, scales = "free", ncol = 2) +
  scale_y_continuous(labels = comma) +
  labs(title = "Amazon Sales Signals: log(Price) and Star Rating",
       subtitle = "Stars cluster near 4.5 — selection bias in voluntary rating systems",
       x = NULL, y = "Count") +
  theme_minimal()

# Top 15 categories by Best Seller share
cat_share <- products %>%
  group_by(category_name) %>%
  summarise(n = n(), share_bs = mean(isBestSeller), .groups = "drop") %>%
  filter(n >= 200) %>%
  slice_max(share_bs, n = 15)

ggplot(cat_share, aes(reorder(category_name, share_bs), share_bs)) +
  geom_segment(aes(xend = category_name, y = 0, yend = share_bs), color = "grey60") +
  geom_point(size = 3, color = "tomato") +
  coord_flip() +
  scale_y_continuous(labels = percent) +
  labs(title = "Amazon Sales Signals: Best Seller Density by Category",
       subtitle = "Tools & Home Improvement leads at 15%+ — 10x higher than Apparel",
       x = NULL, y = "Best Seller Share") +
  theme_minimal()

# Star rating by class
products %>%
  mutate(label = ifelse(isBestSeller == 1, "Best Seller", "Other")) %>%
  ggplot(aes(label, stars, fill = label)) +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(title = "Amazon Sales Signals: Star Rating by Best Seller Status",
       subtitle = "Best Sellers cluster near 5.0 but overlap with non-Best Sellers is large",
       x = NULL, y = "Stars") +
  theme_minimal() +
  theme(legend.position = "none")

# Unconditional star distribution
ggplot(products, aes(stars)) +
  geom_histogram(bins = 30, fill = "gold", color = "grey30") +
  scale_y_continuous(labels = comma) +
  labs(title = "Amazon Sales Signals: Unconditional Star Rating Distribution",
       subtitle = "Non-Gaussian shape — rules out a linear working model for stars",
       x = "Star Rating", y = "Count") +
  theme_minimal()

# Top 10 categories by product count
top_count <- products %>%
  count(category_name, sort = TRUE) %>%
  slice_head(n = 10)

ggplot(top_count, aes(reorder(category_name, n), n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  scale_y_continuous(labels = comma) +
  labs(title = "Amazon Sales Signals: Top 10 Categories by Product Count",
       subtitle = "Apparel dominates volume but not Best Seller density",
       x = NULL, y = "Number of Products") +
  theme_minimal()

# PCA on numeric predictors
num_mat  <- products %>% select(log_price, log_reviews, stars) %>% scale()
pca      <- prcomp(num_mat, center = FALSE, scale. = FALSE)
var_expl <- pca$sdev^2 / sum(pca$sdev^2)

set.seed(1)
plot_idx <- c(
  which(products$isBestSeller == 1),
  sample(which(products$isBestSeller == 0), size = 0.02 * sum(products$isBestSeller == 0))
)

tibble(
  PC1 = pca$x[plot_idx, 1],
  PC2 = pca$x[plot_idx, 2],
  cls = factor(products$isBestSeller[plot_idx], labels = c("Other", "Best Seller"))
) %>%
  arrange(cls) %>%
  ggplot(aes(PC1, PC2, color = cls, alpha = cls)) +
  geom_point(size = 0.7) +
  scale_color_manual(values = c("grey70", "tomato")) +
  scale_alpha_manual(values = c(0.3, 0.9)) +
  labs(title = "Amazon Sales Signals: PCA on Numeric Predictors",
       subtitle = "Best Sellers (red) are not linearly separable — category signal is essential",
       color = NULL, alpha = NULL,
       x = sprintf("PC1 (%.1f%%)", 100 * var_expl[1]),
       y = sprintf("PC2 (%.1f%%)", 100 * var_expl[2])) +
  theme_minimal()

# -----------------------------------------------------------------------------
# 3. Train / Test Split & Class Weights
# -----------------------------------------------------------------------------
set.seed(1)
idx_pos   <- which(products$isBestSeller == 1)
idx_neg   <- which(products$isBestSeller == 0)
train_idx <- c(
  sample(idx_pos, size = floor(0.7 * length(idx_pos))),
  sample(idx_neg, size = floor(0.7 * length(idx_neg)))
)
train <- products[train_idx, ]
test  <- products[-train_idx, ]

w_train <- ifelse(
  train$isBestSeller == 1,
  1 / mean(train$isBestSeller),
  1 / (1 - mean(train$isBestSeller))
)

y_train <- train$isBestSeller
y_test  <- test$isBestSeller

# -----------------------------------------------------------------------------
# 4. Model 1a: Logistic Lasso — Fine-Grained Category (248 categories)
# -----------------------------------------------------------------------------
X_full  <- sparse.model.matrix(
  ~ log_price + log_reviews + stars + category_name - 1,
  data = products
)
X_train <- X_full[train_idx,  , drop = FALSE]
X_test  <- X_full[-train_idx, , drop = FALSE]

set.seed(1)
cv_lasso <- cv.glmnet(
  X_train, y_train,
  family       = "binomial",
  weights      = w_train,
  type.measure = "auc",
  nfolds       = 10
)
plot(cv_lasso)
title("Amazon Sales Signals — Lasso (248 Categories): CV-AUC vs. log(λ)", line = 2.6)

p_lasso <- as.numeric(predict(cv_lasso, X_test, s = "lambda.min", type = "response"))

# Top 10 non-zero coefficients
coef(cv_lasso, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column("term") %>%
  rename(coef = s1) %>%
  filter(term != "(Intercept)", coef != 0) %>%
  arrange(desc(abs(coef))) %>%
  head(10)

# -----------------------------------------------------------------------------
# 5. Model 1b: Logistic Lasso — Coarse Category (18 big_category groups)
# -----------------------------------------------------------------------------
X_full_big  <- sparse.model.matrix(
  ~ log_price + log_reviews + stars + big_category - 1,
  data = products
)
X_train_big <- X_full_big[train_idx,  , drop = FALSE]
X_test_big  <- X_full_big[-train_idx, , drop = FALSE]

set.seed(1)
cv_lasso_big <- cv.glmnet(
  X_train_big, y_train,
  family       = "binomial",
  weights      = w_train,
  type.measure = "auc",
  nfolds       = 10
)
plot(cv_lasso_big)
title("Amazon Sales Signals — Lasso (18 Categories): CV-AUC vs. log(λ)", line = 2.6)

p_lasso_big <- as.numeric(predict(cv_lasso_big, X_test_big,
                                  s = "lambda.min", type = "response"))

# All non-zero coefficients
coef(cv_lasso_big, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column("term") %>%
  rename(coef = s1) %>%
  filter(term != "(Intercept)", coef != 0) %>%
  arrange(desc(abs(coef)))

# -----------------------------------------------------------------------------
# 6. Model 2: Generalized Additive Model
# -----------------------------------------------------------------------------
gam_fit <- gam(
  isBestSeller ~ s(log_reviews) + s(log_price) + stars + big_category,
  family  = binomial,
  data    = train,
  method  = "REML",
  weights = w_train
)
summary(gam_fit)

# GAM partial effects
par(mfrow = c(1, 2))
plot(gam_fit, select = 1, shade = TRUE, seWithMean = TRUE,
     main = "Amazon Sales Signals — GAM: f1(log_reviews) Partial Effect")
plot(gam_fit, select = 2, shade = TRUE, seWithMean = TRUE,
     main = "Amazon Sales Signals — GAM: f2(log_price) Partial Effect")
par(mfrow = c(1, 1))

# EDF table
tibble(
  Smooth    = rownames(summary(gam_fit)$s.table),
  EDF       = summary(gam_fit)$s.table[, "edf"],
  `Ref df`  = summary(gam_fit)$s.table[, "Ref.df"],
  `p-value` = summary(gam_fit)$s.table[, "p-value"]
) %>% as.data.frame()

p_gam <- as.numeric(predict(gam_fit, newdata = test, type = "response"))

# -----------------------------------------------------------------------------
# 7. Model 3: Multilevel Logistic Regression
# -----------------------------------------------------------------------------
set.seed(1)
ml_strat <- train %>%
  mutate(.row = row_number()) %>%
  group_by(category_name) %>%
  slice_sample(n = 200) %>%
  ungroup() %>%
  pull(.row)
ml_pos   <- which(train$isBestSeller == 1)
ml_idx   <- unique(c(ml_strat, ml_pos))
train_ml <- train[ml_idx, ]

ml_fit <- glmer(
  isBestSeller ~ log_price + log_reviews + stars + big_category +
    (1 | category_name),
  data    = train_ml,
  family  = binomial(),
  control = glmerControl(optimizer = "bobyqa",
                         optCtrl   = list(maxfun = 2e5)),
  nAGQ    = 0
)

p_ml <- as.numeric(predict(ml_fit, newdata = test,
                           type = "response", allow.new.levels = TRUE))

# Fixed effects summary
broom.mixed::tidy(ml_fit, effects = "fixed") %>%
  select(term, estimate, std.error, p.value) %>%
  mutate(across(where(is.numeric), ~ round(., 4)))

# Random intercept variance
as.data.frame(VarCorr(ml_fit))

# -----------------------------------------------------------------------------
# 8. Model 4: SVM — RBF Kernel
# -----------------------------------------------------------------------------
set.seed(1)
sub_pos   <- which(train$isBestSeller == 1)
sub_neg   <- sample(which(train$isBestSeller == 0), size = 5 * length(sub_pos))
train_sub <- train[c(sub_pos, sub_neg), ]

sub_prev    <- mean(train_sub$isBestSeller)
svm_weights <- c("0" = 1 / (1 - sub_prev), "1" = 1 / sub_prev)

svm_fit  <- svm(
  factor(isBestSeller) ~ log_price + log_reviews + stars + big_category,
  data          = train_sub,
  kernel        = "radial",
  cost          = 1,
  probability   = TRUE,
  scale         = TRUE,
  class.weights = svm_weights
)
svm_pred <- predict(svm_fit, newdata = test, probability = TRUE)
p_svm    <- attr(svm_pred, "probabilities")[, "1"]

# -----------------------------------------------------------------------------
# 9. Evaluation — ROC and PR Curves
# -----------------------------------------------------------------------------
roc_lasso     <- pROC::roc(y_test, p_lasso,     quiet = TRUE)
roc_lasso_big <- pROC::roc(y_test, p_lasso_big, quiet = TRUE)
roc_gam       <- pROC::roc(y_test, p_gam,       quiet = TRUE)
roc_ml        <- pROC::roc(y_test, p_ml,        quiet = TRUE)
roc_svm       <- pROC::roc(y_test, p_svm,       quiet = TRUE)

pr_lasso     <- pr.curve(scores.class0 = p_lasso[y_test == 1],
                         scores.class1 = p_lasso[y_test == 0],     curve = TRUE)
pr_lasso_big <- pr.curve(scores.class0 = p_lasso_big[y_test == 1],
                         scores.class1 = p_lasso_big[y_test == 0], curve = TRUE)
pr_gam       <- pr.curve(scores.class0 = p_gam[y_test == 1],
                         scores.class1 = p_gam[y_test == 0],       curve = TRUE)
pr_ml        <- pr.curve(scores.class0 = p_ml[y_test == 1],
                         scores.class1 = p_ml[y_test == 0],        curve = TRUE)
pr_svm       <- pr.curve(scores.class0 = p_svm[y_test == 1],
                         scores.class1 = p_svm[y_test == 0],       curve = TRUE)

# AUC summary table
tibble(
  Model     = c("Lasso (small)", "Lasso (big)", "GAM", "Multilevel", "SVM (RBF)"),
  `ROC-AUC` = c(as.numeric(auc(roc_lasso)), as.numeric(auc(roc_lasso_big)),
                as.numeric(auc(roc_gam)),   as.numeric(auc(roc_ml)),
                as.numeric(auc(roc_svm))),
  `PR-AUC`  = c(pr_lasso$auc.integral, pr_lasso_big$auc.integral,
                pr_gam$auc.integral,   pr_ml$auc.integral,
                pr_svm$auc.integral)
) %>% as.data.frame()

# ROC curves plot
roc_df <- bind_rows(
  tibble(Model = "Lasso (small)", FPR = 1 - roc_lasso$specificities,     TPR = roc_lasso$sensitivities),
  tibble(Model = "Lasso (big)",   FPR = 1 - roc_lasso_big$specificities, TPR = roc_lasso_big$sensitivities),
  tibble(Model = "GAM",           FPR = 1 - roc_gam$specificities,       TPR = roc_gam$sensitivities),
  tibble(Model = "Multilevel",    FPR = 1 - roc_ml$specificities,        TPR = roc_ml$sensitivities),
  tibble(Model = "SVM",           FPR = 1 - roc_svm$specificities,       TPR = roc_svm$sensitivities)
)

pr_df <- bind_rows(
  tibble(Model = "Lasso (small)", Recall = pr_lasso$curve[, 1],     Precision = pr_lasso$curve[, 2]),
  tibble(Model = "Lasso (big)",   Recall = pr_lasso_big$curve[, 1], Precision = pr_lasso_big$curve[, 2]),
  tibble(Model = "GAM",           Recall = pr_gam$curve[, 1],       Precision = pr_gam$curve[, 2]),
  tibble(Model = "Multilevel",    Recall = pr_ml$curve[, 1],        Precision = pr_ml$curve[, 2]),
  tibble(Model = "SVM",           Recall = pr_svm$curve[, 1],       Precision = pr_svm$curve[, 2])
)

p_roc <- ggplot(roc_df, aes(FPR, TPR, color = Model)) +
  geom_line(linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  labs(title = "Amazon Sales Signals: ROC Curves",
       subtitle = "Lasso (small) leads with ROC-AUC = 0.8134",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  theme(legend.position = "bottom")

p_pr <- ggplot(pr_df, aes(Recall, Precision, color = Model)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = mean(y_test), linetype = "dashed", color = "grey60") +
  labs(title = "Amazon Sales Signals: Precision-Recall Curves",
       subtitle = "Dashed line = 0.67% random baseline | Best model: 6.5x lift",
       x = "Recall", y = "Precision") +
  theme_minimal() +
  theme(legend.position = "bottom")

grid.arrange(p_roc, p_pr, ncol = 2)

# -----------------------------------------------------------------------------
# 10. Operating-Point Metrics (Top-K)
# -----------------------------------------------------------------------------
rare_event_metrics <- function(p, y, ks = c(0.005, 0.01, 0.05, 0.10)) {
  N         <- length(p)
  total_pos <- sum(y == 1)
  total_neg <- sum(y == 0)
  prev      <- mean(y == 1)
  purrr::map_df(ks, function(k) {
    n_top <- ceiling(k * N)
    thr   <- sort(p, decreasing = TRUE)[n_top]
    flag  <- p >= thr
    tp    <- sum(flag & y == 1)
    tn    <- sum(!flag & y == 0)
    tibble(
      `Top K`     = scales::percent(k, accuracy = 0.1),
      Precision   = tp / max(sum(flag), 1),
      Recall      = tp / total_pos,
      Specificity = tn / total_neg,
      Lift        = (tp / max(sum(flag), 1)) / prev
    )
  })
}

bind_rows(
  rare_event_metrics(p_lasso,     y_test) %>% mutate(Model = "Lasso (small)"),
  rare_event_metrics(p_lasso_big, y_test) %>% mutate(Model = "Lasso (big)"),
  rare_event_metrics(p_gam,       y_test) %>% mutate(Model = "GAM"),
  rare_event_metrics(p_ml,        y_test) %>% mutate(Model = "Multilevel"),
  rare_event_metrics(p_svm,       y_test) %>% mutate(Model = "SVM")
) %>%
  mutate(across(c(Precision, Recall, Specificity), ~ round(., 4)),
         Lift = round(Lift, 1)) %>%
  select(Model, `Top K`, Precision, Recall, Specificity, Lift) %>%
  as.data.frame()