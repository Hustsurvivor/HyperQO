import random
import sys
from ImportantConfig import Config
config = Config()
from sql2fea import TreeBuilder,value_extractor
from NET import TreeNet
from sql2fea import Sql2Vec
from TreeLSTM import SPINN
import torch.nn as nn

sys.stdout = open(config.log_file, "w")
random.seed(113)
with open(config.queries_file) as f:
    qid2query = { line.split('#####')[0] : line.split('#####')[1] for line in f.readlines()}

tree_builder = TreeBuilder()
sql2vec = Sql2Vec()
value_network = SPINN(head_num=config.head_num, input_size=7+2, hidden_size=config.hidden_size, table_num = 50,sql_size = len(config.aliasname2id) ** 2+config.max_column).to(config.device)
for name, param in value_network.named_parameters():
    from torch.nn import init
    if len(param.shape)==2:
        nn.init.xavier_normal_(param)
    else:
        nn.init.uniform_(param)


net = TreeNet(tree_builder= tree_builder,value_network = value_network)
from Hinter import Hinter
from mcts import MCTSHinterSearch

mcts_searcher = MCTSHinterSearch()
hinter = Hinter(model = net,sql2vec = sql2vec,value_extractor = value_extractor,mcts_searcher = mcts_searcher)

print(len(qid2query))
s1 = 0
s2 = 0
s3 = 0
s4 = 0
s_pg = 0
s_hinter = 0
for epoch in range(1):
    for qid, query in qid2query.items():
        print('----',qid,'-----')
        pg_plan_time,pg_latency,mcts_time,hinter_plan_time,MPHE_time,hinter_latency,actual_plans,actual_time = hinter.hinterRun(query)
        pg_latency/=1000
        hinter_latency/=1000
        pg_plan_time/=1000
        hinter_plan_time/=1000
        print('pg plan:',pg_plan_time,'pg run:',pg_latency)
        s1 += pg_plan_time
        print('mcts:',mcts_time,'plan gen:',hinter_plan_time,'MPHE:',MPHE_time,'hinter latency:',hinter_latency)
        s2 += mcts_time
        s3 += hinter_plan_time
        s4 += MPHE_time
        s_pg += pg_latency
        s_hinter += sum(actual_time)/1000
        # print()
        print([actual_plans,actual_time])
        print("%.4f %.4f %.4f %.4f %.4f %.4f %.4f"%(s1,s2,s3,s4,s_pg,s_hinter,s_hinter/s_pg))
        import json
        sys.stdout.flush()
        
    

    