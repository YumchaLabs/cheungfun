# Unity C# 代码索引和问答示例

这个示例展示了如何使用 Cheungfun 框架为 Unity C# 项目构建智能代码索引和问答系统。

## 🎯 功能特性

- **🎮 Unity 项目专用**：自动识别和处理 Unity C# 脚本
- **🌳 AST 增强分割**：基于 tree-sitter 的智能代码分块，理解代码结构
- **📁 Gitignore 支持**：自动遵循项目的 .gitignore 规则，排除不必要的文件
- **🔍 代码结构感知**：理解类、方法、属性等 C# 代码结构
- **💬 智能问答**：支持代码功能、架构、实现细节的自然语言问答
- **📊 详细统计**：提供索引构建和查询的性能统计信息

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保已安装 Rust 和 Cargo
rustc --version
cargo --version

# 克隆 Cheungfun 项目
git clone <cheungfun-repo-url>
cd cheungfun
```

### 2. 设置 API 密钥

#### 使用 OpenAI (推荐)
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### 或使用本地 Ollama
```bash
# 安装并启动 Ollama
ollama serve
ollama pull llama3.2
```

### 3. 运行示例

#### 使用现有 Unity 项目
```bash
# 设置 Unity 项目路径
export UNITY_PROJECT_PATH="/path/to/your/unity/project"

# 运行示例
cargo run --example unity_csharp_rag_example --features "fastembed,sqlite"
```

#### 使用示例项目
```bash
# 不设置路径，系统会自动创建示例项目
cargo run --example unity_csharp_rag_example --features "fastembed"
```

## ⚙️ 配置选项

可以通过环境变量自定义配置：

```bash
# Unity 项目路径
export UNITY_PROJECT_PATH="/path/to/unity/project"

# 代码分块设置
export CHUNK_LINES=30              # 每块代码行数
export CHUNK_LINES_OVERLAP=10      # 块间重叠行数
export MAX_CHARS=1200              # 每块最大字符数

# 检索设置
export TOP_K=5                     # 返回相关代码片段数量

# AST 分析开关
export ENABLE_AST=true             # 启用/禁用 AST 分析
```

## 💬 使用示例

启动系统后，你可以询问各种关于代码的问题：

### 🎮 游戏功能相关
- "这个项目中有哪些主要的游戏对象类？"
- "玩家控制是如何实现的？"
- "游戏中的碰撞检测是怎么处理的？"
- "UI 系统是如何组织的？"

### 🏗️ 架构设计相关
- "项目使用了哪些设计模式？"
- "数据管理是如何实现的？"
- "事件系统是怎么设计的？"
- "场景管理的架构是什么样的？"

### 🔧 技术实现相关
- "这个方法的具体实现逻辑是什么？"
- "如何优化这段代码的性能？"
- "这个类的职责和依赖关系是什么？"
- "错误处理是如何实现的？"

### 🎯 Unity 特定
- "MonoBehaviour 的生命周期是如何使用的？"
- "ScriptableObject 在项目中的应用？"
- "协程的使用场景和实现？"
- "Unity 事件系统的集成方式？"

## 🛠️ 系统命令

在交互模式下，可以使用以下命令：

- `help` - 显示帮助信息
- `examples` - 显示示例问题
- `stats` - 显示系统统计信息
- `quit` 或 `exit` - 退出程序

## 📁 文件过滤规则

系统会自动遵循以下过滤规则：

### ✅ 包含的文件
- `Assets/**/*.cs` - Assets 目录下的所有 C# 文件
- `Scripts/**/*.cs` - Scripts 目录下的所有 C# 文件

### ❌ 排除的文件和目录
- `Library/` - Unity 库文件
- `Temp/` - 临时文件
- `Obj/` - 编译对象文件
- `Build/`, `Builds/` - 构建输出
- `Logs/` - 日志文件
- `UserSettings/` - 用户设置
- `Packages/` - Unity 包管理文件
- `*.csproj`, `*.sln` - 项目文件
- 以及 .gitignore 中定义的其他规则

## 🔧 技术架构

### 核心组件
1. **DirectoryLoader** - 负责加载和过滤 Unity 项目文件
2. **CodeSplitter** - 使用 tree-sitter 进行 AST 增强的代码分割
3. **FastEmbedder** - 生成代码的向量嵌入
4. **InMemoryVectorStore** - 存储和检索向量
5. **SiumaiGenerator** - LLM 生成回答

### AST 分析能力
- 识别 C# 类、方法、属性
- 理解 Unity 特定组件（MonoBehaviour、ScriptableObject）
- 保持代码结构的完整性
- 智能分块，避免破坏语法结构

## 🐛 故障排除

### 常见问题

1. **找不到 C# 文件**
   - 检查 Unity 项目路径是否正确
   - 确认项目中存在 Assets 或 Scripts 目录

2. **API 调用失败**
   - 检查 OpenAI API 密钥是否正确设置
   - 或确认 Ollama 服务是否正常运行

3. **内存不足**
   - 减少 CHUNK_LINES 和 MAX_CHARS 参数
   - 考虑使用 SQLite 存储（需要额外配置）

4. **AST 解析失败**
   - 设置 `ENABLE_AST=false` 禁用 AST 分析
   - 检查 C# 代码语法是否正确

### 调试模式

启用详细日志：
```bash
RUST_LOG=debug cargo run --example unity_csharp_rag_example --features "fastembed"
```

## 📚 扩展开发

这个示例可以作为基础，扩展支持：

- 其他编程语言（JavaScript、Python 等）
- 更复杂的代码分析（依赖关系、调用图等）
- 集成到 IDE 插件中
- 支持更多的向量存储后端
- 添加代码生成和重构建议功能

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个示例！

## 📄 许可证

本示例遵循 Cheungfun 项目的许可证。
