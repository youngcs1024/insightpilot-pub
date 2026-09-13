---
title: GMV（商品交易总额）口径备忘录
doc_type: metric_memo
effective_from: 2025-03-01
effective_to: null
supersedes: null
metric_key: gmv
---

# GMV（商品交易总额）口径备忘录

## 规范定义

指标：gmv；版本：1；名称：GMV（商品交易总额）

业务时区 Asia/Shanghai，期间为半开区间 [start,end)。合格订单已支付、未取消且非测试账号。区域采用下单区域 orders.region_id。按支付时间统计商品金额减优惠，不含运费、不扣退款。品类按商品行 quantity*unit_price-COALESCE(item_discount,0) 汇总，避免订单连接放大；空输入返回 NULL。

默认日期字段：o.paid_at

必需过滤条件：
- o.paid_at IS NOT NULL
- o.status <> 'cancelled'
- c.is_test_account = false

支持粒度：total、day、week、month、region、category

来源表：biz.orders、biz.customers、biz.order_items、biz.products

## 使用说明

金额核对先看是否已经剔除取消订单与测试账号，再核对商品行优惠。GMV 不是到账净收入：运费与退款均不属于本指标的加减项。例如一个订单横跨服饰和家居，按品类拆分时只能分配各自商品行净额，不能在两个品类重复计算整单。没有匹配数据时保留 NULL，并说明范围，而不是补一个看起来完整的零。
