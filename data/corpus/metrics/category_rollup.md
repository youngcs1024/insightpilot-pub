---
title: 品类报表不能随手相加
doc_type: metric_memo
effective_from: 2025-03-01
effective_to: null
supersedes: null
---

# 品类报表不能随手相加

## 一张跨品类订单
一张合格订单可同时包含 apparel 和 home。按品类统计 GMV 时使用各商品行 quantity*unit_price-COALESCE(item_discount,0)，全品类商品净额可核对订单净额。不要把订单总金额连接到每条商品行后直接求和。

| 指标 | 品类汇总注意事项 |
| --- | --- |
| GMV | 各行仅计一次，按实际优惠分摊 |
| order_count | 品类内订单去重，跨品类不可直接相加 |
| active_customer | 品类内按规范化 phone 去重，跨品类不可直接相加 |
| AOV | 先汇总分子分母，再按所问粒度计算 |

## 不支持的维度
当前 refund_rate 和 refund_count 的已发布目录不支持 category 粒度。退款表记录订单退款，不能无依据把每笔款项平均分给订单中所有品类。用户要求品类退款率时，应说明归因资料不足或提出明确的新口径需求，不能生成看似能执行的分摊 SQL 冒充现有指标。

### 对账顺序
先固定时区和半开期间，排除取消与测试账号，再检查连接是否放大。当前 list_price 不能覆盖历史 unit_price。商品数量与订单数量分别展示，防止把多件订单的 GMV 增长误说成客群扩大。
