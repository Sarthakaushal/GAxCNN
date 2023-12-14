from genetic_algo import GAxCNN
from config import GAxCNN as conf
import time

sack = GAxCNN(
        pop_size= 25,
        mutation_param= .1,
        selection= 'pool_selection',
        pooling_type= 'prob',
        max_gen=100
    )

sack.solve()

df_sack = sack.result_to_df()
df_sack.to_csv(conf.output_dir+f"GA_output_{sack.selection}_{str(time.time()).replace('.', '-')}.csv")