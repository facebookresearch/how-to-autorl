import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class IDAAC:
    """
    IDAAC
    """

    def __init__(
        self,
        actor_critic,
        order_classifier,
        clip_param,
        ppo_epoch,
        value_epoch,
        value_freq,
        num_mini_batch,
        value_loss_coef,
        adv_loss_coef,
        order_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
    ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch
        self.value_epoch = value_epoch
        self.value_freq = value_freq
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.adv_loss_coef = adv_loss_coef
        self.order_loss_coef = order_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.policy_parameters = list(actor_critic.base.parameters()) + list(actor_critic.dist.parameters())
        self.value_parameters = list(actor_critic.value_net.parameters())

        self.policy_optimizer = optim.Adam(self.policy_parameters, lr=lr, eps=eps)
        self.value_optimizer = optim.Adam(self.value_parameters, lr=lr, eps=eps)

        self.num_policy_updates = 0

        self.order_classifier = order_classifier
        self.clf_optimizer = optim.Adam(self.order_classifier.parameters(), lr=lr, eps=eps)
        self.bce_loss = nn.BCELoss()

    def update(self, rollouts, device="cpu"):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Update the Policy Network
        order_acc_epoch = 0
        order_loss_epoch = 0
        clf_loss_epoch = 0
        adv_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    obs_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    adv_preds_batch,
                    other_obs_batch,
                    clf_order,
                ) = sample

                (
                    emb,
                    adv,
                    _,
                    action_log_probs,
                    dist_entropy,
                ) = self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                other_emb, _, _, _, _ = self.actor_critic.evaluate_actions(other_obs_batch, actions_batch)

                emb_concat = torch.cat((emb, other_emb), dim=1)
                pred_order_clf = self.order_classifier(emb_concat.detach())
                clf_loss = self.bce_loss(pred_order_clf[:, 0], clf_order.squeeze().float())
                order_acc = (torch.argmax(pred_order_clf, dim=1) == clf_order.squeeze()).sum().item() / len(clf_order)
                self.clf_optimizer.zero_grad()
                clf_loss.backward()
                nn.utils.clip_grad_norm_(self.order_classifier.parameters(), self.max_grad_norm)
                self.clf_optimizer.step()

                pred_order = self.order_classifier(emb_concat)
                emb_order = torch.FloatTensor(size=clf_order.shape).fill_(0.5).to(device)
                order_loss = self.bce_loss(pred_order[:, 0], emb_order.squeeze())

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                adv_loss = (adv - adv_targ).pow(2).mean()

                # Update actor-critic using both PPO Loss
                self.policy_optimizer.zero_grad()
                (
                    order_loss * self.order_loss_coef
                    + adv_loss * self.adv_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(self.policy_parameters, self.max_grad_norm)
                self.policy_optimizer.step()

                order_acc_epoch += order_acc
                order_loss_epoch += order_loss.item()
                clf_loss_epoch += clf_loss.item()
                adv_loss_epoch += adv_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_policy_updates = self.ppo_epoch * self.num_mini_batch

        order_acc_epoch /= num_policy_updates
        order_loss_epoch /= num_policy_updates
        clf_loss_epoch /= num_policy_updates
        adv_loss_epoch /= num_policy_updates
        action_loss_epoch /= num_policy_updates
        dist_entropy_epoch /= num_policy_updates

        # Update the Value Netowrk
        if self.num_policy_updates % self.value_freq == 0:
            value_loss_epoch = 0
            for e in range(self.value_epoch):
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

                for sample in data_generator:
                    (
                        obs_batch,
                        actions_batch,
                        value_preds_batch,
                        return_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                        adv_preds_batch,
                        other_obs_batch,
                        order,
                    ) = sample

                    _, _, values, _, _ = self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                    # Update actor-critic using both PPO Loss
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value_parameters, self.max_grad_norm)
                    self.value_optimizer.step()

                    value_loss_epoch += value_loss.item()

            num_value_updates = self.value_epoch * self.num_mini_batch
            value_loss_epoch /= num_value_updates
            self.prev_value_loss_epoch = value_loss_epoch

        else:
            value_loss_epoch = self.prev_value_loss_epoch

        self.num_policy_updates += 1
        return (
            order_acc_epoch,
            order_loss_epoch,
            clf_loss_epoch,
            adv_loss_epoch,
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
        )
