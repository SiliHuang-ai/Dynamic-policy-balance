import time
import os
import wandb
import numpy as np
from functools import reduce
import torch
from base_runner import Runner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        self.win_rates = []
        self.episode_rewards = []
        self.training_env_steps = []
        self.trained_steps = []
        self.num_agents = config['num_agents']
        self.episode_min_probs = [[] for i in range(self.num_agents)]
        self.episode_max_probs = [[] for i in range(self.num_agents)]
        self.episode_mean_probs = [[] for i in range(self.num_agents)]
        self.cut_probs = []
        self.max_probs = []
        self.plt_dir = str(config["run_dir"] / 'plt')
        if not os.path.exists(self.plt_dir):
            os.makedirs(self.plt_dir)
        super(SMACRunner, self).__init__(config)

    def reset_run(self):
        self.reset_policy()
        self.win_rates = []
        self.episode_rewards = []
        self.training_env_steps = []
        self.episode_min_probs = [[] for i in range(self.num_agents)]
        self.episode_max_probs = [[] for i in range(self.num_agents)]
        self.episode_mean_probs = [[] for i in range(self.num_agents)]
        self.cut_probs = []
        self.max_probs = []
        self.warmup()


    def run(self, num=0):
        self.reset_run()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        evl_tmp = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            self.buffer.refresh_act_probs()
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic,cur_acts_probs,last_cur_acts_probs = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic,cur_acts_probs,last_cur_acts_probs

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Num {} Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(num,
                                self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)

                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape))


            # eval
            if total_num_steps // self.eval_interval >= evl_tmp:
                evl_tmp += 1
                self.eval(total_num_steps, num)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))

        cur_acts_probs = self.trainer.policy.get_probs(np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]))

        last_cur_acts_probs = self.trainer.policy.get_probs(np.concatenate(self.last_buffer[0][step]),
                                              np.concatenate(self.last_buffer[1][step]),
                                              np.concatenate(self.last_buffer[2][step]),
                                              np.concatenate(self.last_buffer[3][step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        cur_acts_probs = cur_acts_probs.detach()
        last_cur_acts_probs = last_cur_acts_probs.detach()
        cur_acts_probs = np.array(np.split(_t2n(cur_acts_probs), self.n_rollout_threads))
        last_cur_acts_probs = np.array(np.split(_t2n(last_cur_acts_probs), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic,cur_acts_probs,last_cur_acts_probs

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic,\
        cur_acts_probs,last_cur_acts_probs = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions,
                           cur_acts_probs, last_cur_acts_probs)

    
    @torch.no_grad()
    def eval(self, total_num_steps,num):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        step_probs, eps_probs = [],[]
        while True:
            self.trainer.prep_rollout()
            eval_actions,action_log_probs, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            #TODO save prob
            act_probs = torch.exp(action_log_probs).to('cpu')
            act_probs = np.array(act_probs)
            step_probs.append(act_probs)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            action_probs = np.array(np.split(_t2n(torch.exp(action_log_probs)), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eps_probs.append(step_probs)
                    step_probs = []
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}
                eval_win_rate = eval_battles_won/eval_episode
                self.win_rates.append(eval_win_rate)
                self.episode_rewards.append(np.mean(eval_episode_rewards))
                self.training_env_steps.append(total_num_steps)
                self.cal_prob(eps_probs)
                self.plt(num)
                self.plt_prob(num)
                print("eval win rate is {}.".format(eval_win_rate))
                #save model
                self.save()

                break

    def cal_prob(self,probs):
        for agent in range(self.num_agents):
            agent_min,agent_max,agent_mean = 1,0,0
            for epi in range(self.all_args.eval_episodes):
                add,mean = 0,0
                for step in range(len(probs[epi])):
                    tmp = probs[epi][step][agent][0]
                    if tmp<1 and tmp>0:
                        add += tmp
                    else:
                        mean = add/(step+1)#TODO+1
                        agent_mean = agent_mean*epi/(epi+1)+mean/(epi+1)
                        continue
                if mean==0:
                    mean = add / (step+1)
                    agent_mean = agent_mean * epi / (epi + 1) + mean / (epi + 1)
                agent_min = mean if mean < agent_min else agent_min
                agent_max = mean if mean > agent_max else agent_max
            self.episode_min_probs[agent].append(agent_min)
            self.episode_max_probs[agent].append(agent_max)
            self.episode_mean_probs[agent].append(agent_mean)
        agent_min, agent_max = 1, 0
        for agent in range(self.num_agents):
            mean = self.episode_mean_probs[agent][-1]
            agent_min = mean if mean < agent_min else agent_min
            agent_max = mean if mean > agent_max else agent_max
        self.cut_probs.append(agent_max-agent_min)
        self.max_probs.append(agent_max)

    def plt(self, num):
            plt.figure()
            plt.cla()
            plt.subplot(2, 1, 1)
            plt.plot(self.training_env_steps, self.win_rates)
            plt.xlabel('step')
            plt.ylabel('win_rate')

            plt.subplot(2, 1, 2)
            plt.plot(self.training_env_steps, self.episode_rewards)
            plt.xlabel('step')
            plt.ylabel('episode_rewards')


            plt.savefig(self.plt_dir + '/plt_{}.png'.format(num), format='png')
            np.save(self.plt_dir + '/win_rates_{}'.format(num), self.win_rates)
            np.save(self.plt_dir + '/episode_rewards_{}'.format(num), self.episode_rewards)
            np.save(self.plt_dir + '/env_step_{}'.format(num), self.training_env_steps)
            np.save(self.plt_dir + '/trained_steps_{}'.format(num), self.trained_steps)

    def plt_prob(self, num):

        plt.figure()
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(self.training_env_steps, self.cut_probs)
        plt.xlabel('step')
        plt.ylabel('cut_probs')

        plt.subplot(2, 1, 2)
        plt.plot(self.training_env_steps, self.max_probs)
        plt.xlabel('step')
        plt.ylabel('max_probs')
        plt.savefig(self.plt_dir + '/plt_probs_{}.png'.format(num), format='png')
        np.save(self.plt_dir + '/cut_probs{}'.format(num), self.cut_probs)
        np.save(self.plt_dir + '/max_probs{}'.format(num), self.max_probs)
