import os
import torch
import imageio
import networkx as nx
import numpy as np
# import rdkit.Chem
import wandb
import matplotlib.pyplot as plt
# import igraph
# import pygraphviz as pgv
import datasets_nas
from configs.ckpt import DATAROOT_NB201


class ArchVisualization:
    def __init__(self, config, remove_none=False, exp_name=None):
        self.config = config
        self.remove_none = remove_none
        self.exp_name = exp_name
        self.num_graphs_to_visualize = config.log.num_graphs_to_visualize
        self.nasbench201 = torch.load(DATAROOT_NB201)
        
        self.labels = {
            0: 'input',
            1: 'output',
            2: 'conv3',
            3: 'sep3',
            4: 'conv5',
            5: 'sep5',
            6: 'avg3',
            7: 'max3',
        }
        
        self.colors = ['skyblue', 'pink', 'yellow', 'orange', 'greenyellow', 'green', 'azure', 'beige']
        

    def to_networkx_directed(self, node_list, adjacency_matrix):
        """
        Convert graphs to neural architectures 
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        
        
        graph = nx.DiGraph()
        # add nodes to the graph
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(torch.triu(torch.tensor(adjacency_matrix), diagonal=1).numpy() >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(edge[0], edge[1], color=float(edge_type), weight=3 * edge_type)
        
        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=1200, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")
            # pos = nx.multipartite_layout(graph, subset_key='number')
            # pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the operations

        plt.figure()
        nx.draw(graph, pos=pos, labels=self.labels, arrows=True, node_shape="s", 
                node_size=node_size, node_color=self.colors, edge_color='grey', with_labels=True)
        # nx.draw(graph, pos, font_size=5, node_size=node_size, with_labels=False, node_color=U[:, 1],
        #         cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')
        # import pdb; pdb.set_trace()
        # plt.tight_layout()
        
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, log='graph', adj=None):
        # define path to save figures
        os.makedirs(path, exist_ok=True)

        # visualize the final molecules
        for i in range(self.num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            graph = self.to_networkx_directed(graphs[i], adj[0].detach().cpu().numpy())
            self.visualize_non_molecule(graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, sample_list, adjacency_matrix,
                        r_valid_chain, r_uniqueness_chain, r_novel_chain):
        import pdb; pdb.set_trace()
        # convert graphs to networkx
        graphs = [self.to_networkx_directed(sample_list[i], adjacency_matrix[i]) for i in range(sample_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.nx_pydot.graphviz_layout(final_graph, prog="dot")
        # final_pos = None

        # draw gif
        save_paths = []
        num_frams = sample_list

        for frame in range(num_frams):
            file_name = os.path.join(path, 'frame_{}.png'.format(frame))
            self.visualize_non_molecule(graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        print(f'==> Save gif at {gif_path}')
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, fps=5)
        if wandb.run:
            wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
    
    
    def visualize_chain_vun(self, path, r_valid_chain, r_unique_chain, r_novel_chain, sde, sampling_eps, number_chain_steps=None):
        
        os.makedirs(path, exist_ok=True)
        # timesteps = torch.linspace(sampling_eps, sde.T, sde.N)
        timesteps = torch.linspace(sde.T, sampling_eps, sde.N)
        
        if number_chain_steps is not None:
            timesteps_ = []
            n = int(sde.N / number_chain_steps)
            for i, t in enumerate(timesteps):
                if i % n == n - 1:
                    timesteps_.append(t.item())
            # timesteps_ = [t for i, t in enumerate(timesteps) if i % n == n-1]
            assert len(timesteps_) == number_chain_steps
            timesteps_ = timesteps_[::-1]
        
        else:
            timesteps_ = list(timesteps.numpy())[::-1]
        
        # validity
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(timesteps_, r_valid_chain, color='red')
        ax.set_title(f'Validity')
        ax.set_xlabel('time')
        ax.set_ylabel('Validity')
        plt.show()
        file_path = os.path.join(path, 'validity.png')
        plt.savefig(file_path)
        plt.close("all")
        print(f'==> Save scatter plot at {file_path}')
        im = plt.imread(file_path)
        if wandb.run:
            wandb.log({'r_valid_chains': [wandb.Image(im, caption=file_path)]})
            
        # Uniqueness
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(timesteps_, r_unique_chain, color='green')
        ax.set_title(f'Uniqueness')
        ax.set_xlabel('time')
        ax.set_ylabel('Uniqueness')
        plt.show()
        file_path = os.path.join(path, 'uniquness.png')
        plt.savefig(file_path)
        plt.close("all")
        print(f'==> Save scatter plot at {file_path}')
        im = plt.imread(file_path)
        if wandb.run:
            wandb.log({'r_uniqueness_chains': [wandb.Image(im, caption=file_path)]})
        
        # Novelty
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(timesteps_, r_novel_chain, color='blue')
        ax.set_title(f'Novelty')
        ax.set_xlabel('time')
        ax.set_ylabel('Novelty')
        file_path = os.path.join(path, 'novelty.png')
        plt.savefig(file_path)
        plt.close("all")
        print(f'==> Save scatter plot at {file_path}')
        im = plt.imread(file_path)
        if wandb.run:
            wandb.log({'r_novelty_chains': [wandb.Image(im, caption=file_path)]})
    
    
    def visualize_grad_norm(self, path, score_grad_norm_p, classifier_grad_norm_p, 
                            score_grad_norm_c, classifier_grad_norm_c, sde, sampling_eps, 
                            number_chain_steps=None):
        
        os.makedirs(path, exist_ok=True)
        # timesteps = torch.linspace(sampling_eps, sde.T, sde.N)
        timesteps = torch.linspace(sde.T, sampling_eps, sde.N)
        timesteps_ = list(timesteps.numpy())[::-1]
        
        if len(score_grad_norm_c) == 0:
            score_grad_norm_c = [-1] * len(score_grad_norm_p)
        if len(classifier_grad_norm_c) == 0:
            classifier_grad_norm_c = [-1] * len(classifier_grad_norm_p)
        
        plt.clf()
        fig, ax1 = plt.subplots()
        
        color_1 = 'red'
        ax1.set_title(f'grad_norm (predictor)')
        ax1.set_xlabel('time')
        ax1.set_ylabel('score_grad_norm (predictor)', color=color_1)
        ax1.plot(timesteps_, score_grad_norm_p, color=color_1)
        ax1.tick_params(axis='y', labelcolor=color_1)
        
        ax2 = ax1.twinx()
        color_2 = 'blue'
        ax2.set_ylabel('classifier_grad_norm (predictor)', color=color_2)
        ax2.plot(timesteps_, classifier_grad_norm_p, color=color_2)
        ax2.tick_params(axis='y', labelcolor=color_2)
        fig.tight_layout()
        plt.show()
        
        file_path = os.path.join(path, 'grad_norm_p.png')
        plt.savefig(file_path)
        plt.close("all")
        print(f'==> Save scatter plot at {file_path}')
        im = plt.imread(file_path)
        if wandb.run:
            wandb.log({'grad_norm_p': [wandb.Image(im, caption=file_path)]})
        
        
        plt.clf()
        fig, ax1 = plt.subplots()
        
        color_1 = 'green'
        ax1.set_title(f'grad_norm (corrector)')
        ax1.set_xlabel('time')
        ax1.set_ylabel('score_grad_norm (corrector)', color=color_1)
        ax1.plot(timesteps_, score_grad_norm_c, color=color_1)
        ax1.tick_params(axis='y', labelcolor=color_1)
        
        ax2 = ax1.twinx()
        color_2 = 'yellow'
        ax2.set_ylabel('classifier_grad_norm (corrector)', color=color_2)
        ax2.plot(timesteps_, classifier_grad_norm_c, color=color_2)
        ax2.tick_params(axis='y', labelcolor=color_2)
        fig.tight_layout()
        plt.show()
        
        file_path = os.path.join(path, 'grad_norm_c.png')
        plt.savefig(file_path)
        plt.close("all")
        print(f'==> Save scatter plot at {file_path}')
        im = plt.imread(file_path)
        if wandb.run:
            wandb.log({'grad_norm_c': [wandb.Image(im, caption=file_path)]})
    
    
    def visualize_scatter(self, path, 
                          score_config, classifier_config, 
                          sampled_arch_metric, plot_textstr=True,
                          x_axis='latency', y_axis='test-acc', x_label='Latency (ms)', y_label='Accuracy (%)',
                          log='scatter', check_dataname='cifar10-valid',
                          selected_arch_idx_list_topN=None, selected_arch_idx_list=None,
                          train_idx_list=None, return_file_path=False):
        
        os.makedirs(path, exist_ok=True)
        
        tg_dataset = classifier_config.data.tg_dataset
        
        train_ds_s, eval_ds_s, test_ds_s = datasets_nas.get_dataset(score_config)
        if selected_arch_idx_list is None:
            train_ds_c, eval_ds_c, test_ds_c = datasets_nas.get_dataset(classifier_config)
        else:
            train_ds_c, eval_ds_c, test_ds_c = datasets_nas.get_dataset_iter(classifier_config)
        
        plt.clf()
        fig, ax = plt.subplots()
        
        # entire architectures
        entire_ds_x = train_ds_s.get_unnoramlized_entire_data(x_axis, tg_dataset)
        entire_ds_y = train_ds_s.get_unnoramlized_entire_data(y_axis, tg_dataset)
        ax.scatter(entire_ds_x, entire_ds_y, color = 'lightgray', alpha = 0.5, label='Entire', marker=',')
        
        # architectures trained by the score_model
        # train_ds_s_x = train_ds_s.get_unnoramlized_data(x_axis, tg_dataset)
        # train_ds_s_y = train_ds_s.get_unnoramlized_data(y_axis, tg_dataset)
        # ax.scatter(train_ds_s_x, train_ds_s_y, color = 'gray', alpha = 0.8, label='Trained by Score Model')
        
        # architectures trained by the classifier
        train_ds_c_x = train_ds_c.get_unnoramlized_data(x_axis, tg_dataset)
        train_ds_c_y = train_ds_c.get_unnoramlized_data(y_axis, tg_dataset)
        ax.scatter(train_ds_c_x, train_ds_c_y, color = 'black', alpha = 0.8, label='Trained by Predictor Model')
        
        # oracle
        oracle_idx = torch.argmax(torch.tensor(entire_ds_y)).item()
        # oracle_idx = torch.argmax(torch.tensor(train_ds_s.get_unnoramlized_entire_data('val-acc', tg_dataset))).item()
        oracle_item_x = entire_ds_x[oracle_idx]
        oracle_item_y = entire_ds_y[oracle_idx]
        ax.scatter(oracle_item_x, oracle_item_y, color = 'red', alpha = 1.0, label='Oracle', marker='*', s=150)
        
        # architectures sampled by the score_model & classifier
        AXIS_TO_PROP = {
            'val-acc': 'val_acc_list',
            'test-acc': 'test_acc_list',
            'latency': 'latency_list',
            'flops': 'flops_list',
            'params': 'params_list',
        }
        sampled_ds_c_x = sampled_arch_metric[2][AXIS_TO_PROP[x_axis]]
        sampled_ds_c_y = sampled_arch_metric[2][AXIS_TO_PROP[y_axis]]
        ax.scatter(sampled_ds_c_x, sampled_ds_c_y, color = 'limegreen', alpha = 0.8, label='Sampled',  marker='x')
        
        ax.set_title(f'{tg_dataset.upper()} Dataset')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        

        if selected_arch_idx_list_topN is not None:
            selected_arch_topN_info_dict = get_arch_acc_info_dict(
                self.nasbench201, dataname=check_dataname, arch_index_list=selected_arch_idx_list_topN)
            selected_topN_ds_x = selected_arch_topN_info_dict[AXIS_TO_PROP[x_axis]]
            selected_topN_ds_y = selected_arch_topN_info_dict[AXIS_TO_PROP[y_axis]]
            ax.scatter(selected_topN_ds_x, selected_topN_ds_y, color = 'pink', alpha = 0.8, label='Selected_topN',  marker='x')
        
        # architectures selected by the prdictor
        selected_ds_x, selected_ds_y = None, None
        if selected_arch_idx_list is not None:
            selected_arch_info_dict = get_arch_acc_info_dict(
                self.nasbench201, dataname=check_dataname, arch_index_list=selected_arch_idx_list)
            selected_ds_x = selected_arch_info_dict[AXIS_TO_PROP[x_axis]]
            selected_ds_y = selected_arch_info_dict[AXIS_TO_PROP[y_axis]]
            ax.scatter(selected_ds_x, selected_ds_y, color = 'blue', alpha = 0.8, label='Selected',  marker='x')
        
        if plot_textstr:
            textstr = self.get_textstr(sampled_arch_metric=sampled_arch_metric, 
                                       sampled_ds_c_x=sampled_ds_c_x, sampled_ds_c_y=sampled_ds_c_y, 
                                       x_axis=x_axis, y_axis=y_axis, 
                                       classifier_config=classifier_config, 
                                       selected_ds_x=selected_ds_x, selected_ds_y=selected_ds_y, 
                                       selected_topN_ds_x=selected_topN_ds_x, selected_topN_ds_y=selected_topN_ds_y,
                                       oracle_idx=oracle_idx, train_idx_list=train_idx_list
                                       )
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.6, 0.4, textstr, transform=ax.transAxes, verticalalignment='bottom', bbox=props, fontsize='x-small')
            # ax.text(textstr, transform=ax.transAxes, verticalalignment='bottom', bbox=props)
            ax.legend(loc="lower right")
        
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        plt.show()
        plt.tight_layout()
        
        file_path = os.path.join(path, 'scatter.png')
        plt.savefig(file_path)
        plt.close("all")
        print(f'==> Save scatter plot at {path}')
        
        if return_file_path:
            return file_path
        
        im = plt.imread(file_path)
        if wandb.run and log is not None:
            wandb.log({log: [wandb.Image(im, caption=file_path)]})
        
        # if return_selected_arch_info_dict:
        #     return selected_arch_info_dict, selected_arch_topN_info_dict
    
    def visualize_scatter_chain(self, path, score_config, classifier_config, sampled_arch_metric_chain, plot_textstr=True,
                          x_axis='latency', y_axis='test-acc', x_label='Latency (ms)', y_label='Accuracy (%)',
                          log='scatter_chain'):
        
        # draw gif
        os.makedirs(path, exist_ok=True)
        save_paths = []
        num_frames = len(sampled_arch_metric_chain)
        
        tg_dataset = classifier_config.data.tg_dataset
        
        train_ds_s, eval_ds_s, test_ds_s = datasets_nas.get_dataset(score_config)
        train_ds_c, eval_ds_c, test_ds_c = datasets_nas.get_dataset(classifier_config)
        
        # entire architectures
        entire_ds_x = train_ds_s.get_unnoramlized_entire_data(x_axis, tg_dataset)
        entire_ds_y = train_ds_s.get_unnoramlized_entire_data(y_axis, tg_dataset)
        
        # architectures trained by the score_model
        train_ds_s_x = train_ds_s.get_unnoramlized_data(x_axis, tg_dataset)
        train_ds_s_y = train_ds_s.get_unnoramlized_data(y_axis, tg_dataset)
        
        # architectures trained by the classifier
        train_ds_c_x = train_ds_c.get_unnoramlized_data(x_axis, tg_dataset)
        train_ds_c_y = train_ds_c.get_unnoramlized_data(y_axis, tg_dataset)
        
        # oracle
        # oracle_idx = torch.argmax(torch.tensor(entire_ds_y)).item()
        oracle_idx = torch.argmax(torch.tensor(train_ds_s.get_unnoramlized_entire_data('val-acc', tg_dataset))).item()
        oracle_item_x = entire_ds_x[oracle_idx]
        oracle_item_y = entire_ds_y[oracle_idx]
    
        for frame in range(num_frames):
            sampled_arch_metric = sampled_arch_metric_chain[frame]
            
            plt.clf()
            fig, ax = plt.subplots()
            
            # entire architectures
            ax.scatter(entire_ds_x, entire_ds_y, color = 'lightgray', alpha = 0.5, label='Entire', marker=',')
            # architectures trained by the score_model
            ax.scatter(train_ds_s_x, train_ds_s_y, color = 'gray', alpha = 0.8, label='Trained by Score Model')
            # architectures trained by the classifier
            ax.scatter(train_ds_c_x, train_ds_c_y, color = 'black', alpha = 0.8, label='Trained by Predictor Model')
            # oracle
            ax.scatter(oracle_item_x, oracle_item_y, color = 'red', alpha = 1.0, label='Oracle', marker='*', s=150)
            # architectures sampled by the score_model & classifier
            AXIS_TO_PROP = {
                'test-acc': 'test_acc_list',
                'latency': 'latency_list',
                'flops': 'flops_list',
                'params': 'params_list',
            }
            sampled_ds_c_x = sampled_arch_metric[2][AXIS_TO_PROP[x_axis]]
            sampled_ds_c_y = sampled_arch_metric[2][AXIS_TO_PROP[y_axis]]
            ax.scatter(sampled_ds_c_x, sampled_ds_c_y, color = 'limegreen', alpha = 0.8, label='Sampled',  marker='x')
            
            ax.set_title(f'{tg_dataset.upper()} Dataset')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if plot_textstr:
                textstr = self.get_textstr(sampled_arch_metric, sampled_ds_c_x, sampled_ds_c_y, 
                                           x_axis, y_axis, classifier_config)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.6, 0.3, textstr, transform=ax.transAxes, verticalalignment='bottom', bbox=props)
                # ax.text(textstr, transform=ax.transAxes, verticalalignment='bottom', bbox=props)
                ax.legend(loc="lower right")
            
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
            plt.show()
            # plt.tight_layout()
            
            file_path = os.path.join(path, f'frame_{frame}.png')
            plt.savefig(file_path)
            plt.close("all")
            print(f'==> Save scatter plot at {file_path}')
            save_paths.append(file_path)
            
            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})
        
        # draw gif
        imgs = [imageio.imread(fn) for fn in save_paths[::-1]]
        # gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        gif_path = os.path.join(path, f'scatter.gif')
        print(f'==> Save gif at {gif_path}')
        imgs.extend([imgs[-1]] * 10)
        # imgs.extend([imgs[0]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, fps=5)
        if wandb.run:
            wandb.log({'chain_gif': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
    
    def get_textstr(self, 
                    sampled_arch_metric, 
                    sampled_ds_c_x, sampled_ds_c_y, 
                    x_axis='latency', y_axis='test-acc', 
                    classifier_config=None,
                    selected_ds_x=None, selected_ds_y=None,
                    selected_topN_ds_x=None, selected_topN_ds_y=None,
                    oracle_idx=None, train_idx_list=None):
        mean_v_x = round(np.mean(np.array(sampled_ds_c_x)), 4)
        std_v_x = round(np.std(np.array(sampled_ds_c_x)), 4)
        max_v_x = round(np.max(np.array(sampled_ds_c_x)), 4)
        min_v_x = round(np.min(np.array(sampled_ds_c_x)), 4)
        
        mean_v_y = round(np.mean(np.array(sampled_ds_c_y)), 4)
        std_v_y = round(np.std(np.array(sampled_ds_c_y)), 4)
        max_v_y = round(np.max(np.array(sampled_ds_c_y)), 4)
        min_v_y = round(np.min(np.array(sampled_ds_c_y)), 4)

        if selected_ds_x is not None:
            mean_v_x_s = round(np.mean(np.array(selected_ds_x)), 4)
            std_v_x_s = round(np.std(np.array(selected_ds_x)), 4)
            max_v_x_s = round(np.max(np.array(selected_ds_x)), 4)
            min_v_x_s = round(np.min(np.array(selected_ds_x)), 4)
        
        if selected_ds_y is not None:
            mean_v_y_s = round(np.mean(np.array(selected_ds_y)), 4)
            std_v_y_s = round(np.std(np.array(selected_ds_y)), 4)
            max_v_y_s = round(np.max(np.array(selected_ds_y)), 4)
            min_v_y_s = round(np.min(np.array(selected_ds_y)), 4)
        
        textstr = ''
        r_valid, r_unique, r_novel = round(sampled_arch_metric[0][0], 4), round(sampled_arch_metric[0][1], 4),  round(sampled_arch_metric[0][2], 4)
        textstr += f'V-{r_valid} | U-{r_unique} | N-{r_novel} \n'
        textstr += f'Predictor (Noise-aware-{str(classifier_config.training.noised)[0]}, k={self.config.sampling.classifier_scale}) \n'
        textstr += f'=> Sampled {x_axis} \n'
        textstr += f'Mean-{mean_v_x} | Std-{std_v_x} \n'
        textstr += f'Max-{max_v_x} | Min-{min_v_x} \n'
        textstr += f'=> Sampled {y_axis} \n'
        textstr += f'Mean-{mean_v_y} | Std-{std_v_y} \n'
        textstr += f'Max-{max_v_y} | Min-{min_v_y} \n'
        if selected_ds_x is not None:
            textstr += f'==> Selected {x_axis} \n'
            textstr += f'Mean-{mean_v_x_s} | Std-{std_v_x_s} \n'
            textstr += f'Max-{max_v_x_s} | Min-{min_v_x_s} \n'
        if selected_ds_y is not None:
            textstr += f'==> Selected {y_axis} \n'
            textstr += f'Mean-{mean_v_y_s} | Std-{std_v_y_s} \n'
            textstr += f'Max-{max_v_y_s} | Min-{min_v_y_s} \n'
        if selected_topN_ds_y is not None:
            textstr += f'==> Predicted TopN (10) -{str(round(max(selected_topN_ds_y[:10]), 4))} \n'
        
        if train_idx_list is not None and oracle_idx in train_idx_list:
            textstr += f'==> Hit Oracle ({oracle_idx}) !'
        
        return textstr


def get_arch_acc_info_dict(nasbench201, dataname='cifar10-valid', arch_index_list=None):
    val_acc_list = []
    test_acc_list = []
    flops_list = []
    params_list = []
    latency_list = []
    
    for arch_index in arch_index_list:
        val_acc = nasbench201['val-acc'][dataname][arch_index]
        val_acc_list.append(val_acc)
        test_acc = nasbench201['test-acc'][dataname][arch_index]
        test_acc_list.append(test_acc)
        flops = nasbench201['flops'][dataname][arch_index]
        flops_list.append(flops)
        params = nasbench201['params'][dataname][arch_index]
        params_list.append(params)
        latency = nasbench201['latency'][dataname][arch_index]
        latency_list.append(latency)
    
    return {
        'val_acc_list': val_acc_list,
        'test_acc_list': test_acc_list,
        'flops_list': flops_list,
        'params_list': params_list,
        'latency_list': latency_list
    }