# 2019 China University Computer Contest —— Big Data Challenge 

![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)

2019中国高校计算机大赛——大数据挑战赛

初赛rank12，复赛 A榜 rank15，B榜rank28

[赛题链接](https://www.kesci.com/home/competition/5cc51043f71088002c5b8840)

## Contributor
- [zspo](https://github.com/zspo)
- [Campsa](https://github.com/jinchenyu)

## 关于这次比赛出现的一些问题

这次比赛出现问题比较多，初赛前期没有构建有力的nn模型，最终没能进入前10，复赛B榜最后7天，特征concat出错，w2v增量训练出错再次重跑

1. 没有观察到title字段，test中含有\t，train没有，影响了对数据分布判断
（我一直认为title没有任何重复，这问题从初赛就出现了，队友发现了这个问题并在nn输入模型的时候做了处理，但我一直没发现，在此检讨，没有认真做EDA），
导致count，ctr等统计特征受到较大影响

2. 复赛b榜训练不够，只使用3kw数据训练，和队友沟通不当，导致部分特征使用数据量不同（我偏向于精确到每个query，队友偏向于使用整数），最后concat时数据出现错位， 最终1亿训练集无法使用

3. nn模型上缺乏模型多样性，仅仅使用了简单esim + 手工特征的方式，后期测试不同模型带来提高很大

4. 关于储存，不同比赛有不同打法，初赛最大的瓶颈在于没有GPU，没有算力，传统机器学习模型，相比深度模型有更快的速度，
而在复赛，虽然我们尽可能的在压榨CPU，GPU，尽量同时跑满，但是对硬盘利用仍然停留在csv，早就知道有hdf，但一直没用，直到最后7天储存爆炸，同时浪费了不少读取写入的时间






