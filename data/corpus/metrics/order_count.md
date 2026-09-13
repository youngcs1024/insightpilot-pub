---
title: 支付订单数口径备忘录
doc_type: metric_memo
effective_from: 2025-03-01
effective_to: null
supersedes: null
metric_key: order_count
---

# 支付订单数口径备忘录

## 规范定义

指标：order_count；版本：1；名称：支付订单数

业务时区 Asia/Shanghai，期间为半开区间 [start,end)。合格订单已支付、未取消且非测试账号。区域采用下单区域 orders.region_id。按支付时间统计合格订单去重数；品类内按 order_id 去重，同一订单可属于多个品类，品类不可直接相加。

默认日期字段：o.paid_at

必需过滤条件：
- o.paid_at IS NOT NULL
- o.status <> 'cancelled'
- c.is_test_account = false

支持粒度：total、day、week、month、region、category

来源表：biz.orders、biz.customers、biz.order_items、biz.products

## 使用说明

支付订单数回答发生了多少次合格交易，不回答有多少件商品或多少位客户。连接商品行后仍需按订单去重；同一订单买了多个品类，可以同时进入各品类的去重数，所以各品类相加通常不等于总订单数。跨区域比较使用订单下单区域，不把客户当前区域或发货仓当成销售归属。
