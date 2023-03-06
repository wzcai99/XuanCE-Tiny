# import metaworld
# import random
# import gym
# import gym.spaces
# import numpy as np
# MT10_V2 = ['reach-v2','push-v2','pick-place-v2','door-open-v2','drawer-open-v2','drawer-close-v2','button-press-topdown-v2','peg-insert-side-v2','window-open-v2','window-close-v2']
# # MT50_V2 = [('assembly-v2', SawyerNutAssemblyEnvV2)]
# #     ('basketball-v2', SawyerBasketballEnvV2),
# #     ('bin-picking-v2', SawyerBinPickingEnvV2),
# #     ('box-close-v2', SawyerBoxCloseEnvV2),
# #     ('button-press-topdown-v2', SawyerButtonPressTopdownEnvV2),
# #     ('button-press-topdown-wall-v2', SawyerButtonPressTopdownWallEnvV2),
# #     ('button-press-v2', SawyerButtonPressEnvV2),
# #     ('button-press-wall-v2', SawyerButtonPressWallEnvV2),
# #     ('coffee-button-v2', SawyerCoffeeButtonEnvV2),
# #     ('coffee-pull-v2', SawyerCoffeePullEnvV2),
# #     ('coffee-push-v2', SawyerCoffeePushEnvV2),
# #     ('dial-turn-v2', SawyerDialTurnEnvV2),
# #     ('disassemble-v2', SawyerNutDisassembleEnvV2),
# #     ('door-close-v2', SawyerDoorCloseEnvV2),
# #     ('door-lock-v2', SawyerDoorLockEnvV2),
# #     ('door-open-v2', SawyerDoorEnvV2),
# #     ('door-unlock-v2', SawyerDoorUnlockEnvV2),
# #     ('hand-insert-v2', SawyerHandInsertEnvV2),
# #     ('drawer-close-v2', SawyerDrawerCloseEnvV2),
# #     ('drawer-open-v2', SawyerDrawerOpenEnvV2),
# #     ('faucet-open-v2', SawyerFaucetOpenEnvV2),
# #     ('faucet-close-v2', SawyerFaucetCloseEnvV2),
# #     ('hammer-v2', SawyerHammerEnvV2),
# #     ('handle-press-side-v2', SawyerHandlePressSideEnvV2),
# #     ('handle-press-v2', SawyerHandlePressEnvV2),
# #     ('handle-pull-side-v2', SawyerHandlePullSideEnvV2),
# #     ('handle-pull-v2', SawyerHandlePullEnvV2),
# #     ('lever-pull-v2', SawyerLeverPullEnvV2),
# #     ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
# #     ('pick-place-wall-v2', SawyerPickPlaceWallEnvV2),
# #     ('pick-out-of-hole-v2', SawyerPickOutOfHoleEnvV2),
# #     ('reach-v2', SawyerReachEnvV2),
# #     ('push-back-v2', SawyerPushBackEnvV2),
# #     ('push-v2', SawyerPushEnvV2),
# #     ('pick-place-v2', SawyerPickPlaceEnvV2),
# #     ('plate-slide-v2', SawyerPlateSlideEnvV2),
# #     ('plate-slide-side-v2', SawyerPlateSlideSideEnvV2),
# #     ('plate-slide-back-v2', SawyerPlateSlideBackEnvV2),
# #     ('plate-slide-back-side-v2', SawyerPlateSlideBackSideEnvV2),
# #     ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
# #     ('peg-unplug-side-v2', SawyerPegUnplugSideEnvV2),
# #     ('soccer-v2', SawyerSoccerEnvV2),
# #     ('stick-push-v2', SawyerStickPushEnvV2),
# #     ('stick-pull-v2', SawyerStickPullEnvV2),
# #     ('push-wall-v2', SawyerPushWallEnvV2),
# #     ('push-v2', SawyerPushEnvV2),
# #     ('reach-wall-v2', SawyerReachWallEnvV2),
# #     ('reach-v2', SawyerReachEnvV2),
# #     ('shelf-place-v2', SawyerShelfPlaceEnvV2),
# #     ('sweep-into-v2', SawyerSweepIntoGoalEnvV2),
# #     ('sweep-v2', SawyerSweepEnvV2),
# #     ('window-open-v2', SawyerWindowOpenEnvV2),
# #     ('window-close-v2', SawyerWindowCloseEnvV2),
# # ))

# class MT10_Env(gym.Env):
#     def __init__(self,config,task_index):
#         assert task_index < 10
#         self.task_index = task_index
#         mt1 = metaworld.MT1(MT10_V2[task_index])
#         self.env = mt1.train_classes[MT10_V2[task_index]]()
#         task = random.choice(mt1.train_tasks)
#         self.env.set_task(task)  # Set task
#         self.env._freeze_rand_vec = False
#         self.config = config
#         self.timelimit = config.timelimit
#         self.observation_space = gym.spaces.Dict({'observation':self.env.observation_space,'task_index':gym.spaces.Box(0,10,(1,))})
#         self.action_space = self.env.action_space
#         self._reward_range = (-5,5)
#         self._metadata = {}
#         super(MT10_Env,self).__init__()    
#     def reset(self):
#         self.timesteps = 0
#         obs = self.env.reset()
#         return {'observation':obs,'task_index':np.array([self.task_index])},{}
#     def step(self, action):
#         # revise the correct action range
#         trunction = False
#         obs, reward, done, info = self.env.step(action)
#         # increase the timesteps
#         self.timesteps += 1
#         if self.timesteps >= self.config.timelimit:
#             trunction = True
#         return {'observation':obs,'task_index':np.array([self.task_index])}, reward, done ,trunction, info
#     def render(self):
#         self.env.render()
#     def seed(self,seed:int):
#         self.env.seed(seed)
#     def close(self):
#         self.env.close()


        
