** 分词源码解析
llama_tokenize
分词会创建一个session llm_tokenizer_bpe_session 这里用的是Byte Pair Encoding算法
通过session实现bpe，实现过程：

tokenize
会将词进行连接 比如 you 分成y o u 三个llm_symbol的结构 
然后 对 y o u进行合并 -> yo ou 找到他们的bpe的索引 索引越小表示越早合并、优先级越高
对高优先级的bigram进行合并 最后得到token


rank_found = vocab.find_bpe_rank(left_token, right_token);返回
struct comparator {
    bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
        return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
    }
};

llama_batch_allocr init:
检查batch的有效输入
自动生成缺失的字段：
n_seq_id -> 这个token关联的seq的数量 
seq_id -> 表示第 i 个 token 属于哪些序列
pos -> pos[i] 表示第 i 个 token 在其所属序列中的位置索引（即该 token 在序列里的时间步/位置）
pos没有memory就从0开始 有memory就从上下文的pos+1开始