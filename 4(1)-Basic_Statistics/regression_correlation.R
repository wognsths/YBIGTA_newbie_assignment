library(readr);library(tidyverse);library(data.table)
### Advertising.csv를 불러와 데이터 로드하기!

# ------------------------------------------------------------------------------
# 1. Advertising.csv 불러오기
# ------------------------------------------------------------------------------
Advertising <- read_csv("4(1)-Basic_Statistics/Advertising.csv")
Advertising

### Multiple Linear Regression을 수행해봅시다!

# ------------------------------------------------------------------------------
# 2. Multiple Linear Regression 수행
# ------------------------------------------------------------------------------
#  - 모델: sales ~ TV + radio + newspaper
#  - lm()으로 회귀모델을 적합시킨 뒤 summary()로 요약
# ------------------------------------------------------------------------------

Advertising %>% 
  lm(sales ~ TV + radio + newspaper, data = .) %>%
  summary(.) -> mlr

# ------------------------------------------------------------------------------
# 3. 회귀계수, 표준오차, t-값, p-값 등을 요약 테이블(tb)로 정리
# ------------------------------------------------------------------------------

mlr$coefficients %>%
  as.data.frame() %>%
  rename(
    Coefficients = Estimate,
    `t-statistic` = `t value`,
    `p-value` = `Pr(>|t|)`) %>%
  mutate(
    Coefficients = round(Coefficients, 3),
    `Std. Error` = round(`Std. Error`, 4),
    `t-statistic` = round(`t-statistic`, 2),
    `p-value` = ifelse(`p-value` < 0.0001, "< 0.0001", round(`p-value`, 4))
    ) -> tb

tb

### Correlation Matrix를 만들어 출력해주세요!

# ------------------------------------------------------------------------------
# 4. 사용자 정의 반올림 함수 정의 (Round half away from zero)
# ------------------------------------------------------------------------------
# - pdf에서 제공한 결과값과 달라 rounding 방식 문제라고 생각하여 작성
# - 문제 x : TV와 newspaper의 상관계수는 0.05664788로, 0.0566으로 rounding되는
#   것이 정상임
# ------------------------------------------------------------------------------


round_half_up <- function(x, digits = 0) {
  posneg <- sign(x)
  z <- abs(x) * 10^digits
  z <- floor(z + 0.5)
  z / 10^digits * posneg
}

# ------------------------------------------------------------------------------
# 5. Correlation Matrix 생성 및 하삼각 NA 처리
# ------------------------------------------------------------------------------
#  - Advertising에서 첫 번째 열(sales)을 제외한 나머지로 cor() 계산
#  - 사용자 정의 반올림 함수 적용하여 소수점 처리
#  - lower.tri(...)로 하삼각 구역을 NA로 만들어 표시되지 않게 함
#  - NA를 빈문자로 치환 후, sprintf()로 소수점 자릿수 맞춤
# ------------------------------------------------------------------------------

correlation_table <- Advertising[, -1] %>%
  cor() %>%
  as.data.frame()

correlation_table <- correlation_table %>%
  mutate(across(where(is.numeric), ~ round_half_up(.x, 10)))

correlation_table[lower.tri(correlation_table, diag = FALSE)] <- NA

correlation_table <- correlation_table %>%
  mutate(
    across(
      .cols = where(is.numeric),
      .fns = ~ ifelse(
        is.na(.), 
        "", 
        sprintf("%.4f", .)
      )
    )
  )

correlation_table



