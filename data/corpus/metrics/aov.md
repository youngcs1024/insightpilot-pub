---
title: 客单价口径备忘录
doc_type: metric_memo
effective_from: 2025-03-01
effective_to: null
supersedes: null
metric_key: aov
---

# 客单价口径备忘录

## 规范定义

指标：aov；版本：1；名称：客单价

业务时区 Asia/Shanghai，期间为半开区间 [start,end)。合格订单已支付、未取消且非测试账号。区域采用下单区域 orders.region_id。同粒度 GMV 除以支付订单去重数，零分母返回 NULL。品类客单价为品类商品净额除以包含该品类的订单数。

默认日期字段：o.paid_at

必需过滤条件：
- o.paid_at IS NOT NULL
- o.status <> 'cancelled'
- c.is_test_account = false

支持粒度：total、day、week、month、region、category

来源表：biz.orders、biz.customers、biz.order_items、biz.products

## 使用说明

先固定同一期间和粒度，再计算 GMV 与支付订单数。AOV 是两者的比值，不是各商品单价平均数，也不是多个地区客单价的简单平均。品类口径使用该品类商品净额除以含该品类的合格订单数。缺少分母时报告无法计算；跨月汇总应先合并分子分母再相除，不能平均两个月的 AOV。
