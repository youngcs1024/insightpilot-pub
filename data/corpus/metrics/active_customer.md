---
title: 活跃客户数口径备忘录
doc_type: metric_memo
effective_from: 2025-03-01
effective_to: null
supersedes: null
metric_key: active_customer
---

# 活跃客户数口径备忘录

## 规范定义

指标：active_customer；版本：1；名称：活跃客户数

业务时区 Asia/Shanghai，期间为半开区间 [start,end)。合格订单已支付、未取消且非测试账号。区域采用下单区域 orders.region_id。期间有合格支付订单的客户按规范化非空 phone 去重；这是合成数据集身份约定，不是真实世界身份政策。品类内去重且不可直接相加。

默认日期字段：o.paid_at

必需过滤条件：
- o.paid_at IS NOT NULL
- o.status <> 'cancelled'
- c.is_test_account = false

支持粒度：total、day、week、month、region、category

来源表：biz.orders、biz.customers、biz.order_items、biz.products

## 使用说明

本演示的不同 customer_id 可能共用同一规范化 phone，故按账号数会高估指标。这个约定只属于合成数据集；现实中的家庭共号或号码回收需要额外身份治理。没有支付的注册用户不会进入本指标。客户在两个品类均购买时，两边都可出现，因此品类活跃客户数不可直接相加。
