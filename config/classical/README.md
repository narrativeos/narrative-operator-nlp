# 古汉语领域词典

## 目录结构

```
config/classical/
├── README.md              # 本文件 - 使用说明
├── seed/                  # 开源种子词典 (只读，定期同步)
│   ├── titles.yaml        # 官职名
│   ├── eras.yaml          # 朝代名
│   ├── locations.yaml     # 地名
│   ├── persons.yaml       # 人名
│   ├── classics.yaml      # 典籍名
│   └── institutions.yaml  # 典章制度
├── custom/                # 自维护词典 (可手动编辑)
│   ├── titles.yaml
│   ├── eras.yaml
│   └── ...
└── user/                  # 用户自定义 (运行时注入)
    └── ...
```

## 数据来源

| 文件 | 来源 | 更新方式 |
|------|------|---------|
| seed/titles.yaml | CLNER + 手动整理 | 定期同步 |
| seed/eras.yaml | 手动整理 | 手动更新 |
| seed/locations.yaml | CLNER + DuEE | 定期同步 |
| seed/persons.yaml | CLNER | 定期同步 |
| seed/classics.yaml | 手动整理 | 手动更新 |
| seed/institutions.yaml | 手动整理 | 手动更新 |

## 使用方式

词典在 `EntityMappingRules` 初始化时自动加载。