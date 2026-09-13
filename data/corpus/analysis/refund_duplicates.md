---
title: 部分退款对计数的影响
doc_type: analysis_note
effective_from: 2026-08-01
effective_to: null
supersedes: null
---

# 部分退款对计数的影响

## 容易出现的错觉
一个订单有多次部分退款申请时，直接连接 orders 与 refunds 再统计订单行数会放大订单数量。不能因为 SQL 成功执行，就把结果当作退款订单率。
## 对照口径
默认 refund_rate 的分子按 order_id 去重，排除 rejected 并使用申请期；refund_count 数最终 completed 的退款记录，按 refund_id 计数且仍按申请日归期。二者允许数值不同，不能要求它们在任何报表中相等。
## 报告写法
列出指标名称、期间和观察截止，再说明一张订单可对应多笔记录。请求付款同期群时用同一支付集合做分母，禁止通过缩短或放宽日期把结果改得更像预期。本文不提供当前数据集的实际计数，数值应从执行证据引用。
