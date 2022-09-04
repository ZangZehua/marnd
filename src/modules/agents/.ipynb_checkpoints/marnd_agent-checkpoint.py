import torch
import torch.nn as nn
import torch.nn.functional as F


class MarndAgent(nn.Module):
    """
    input shape: [batch_size * n_agents, input_dim]
    output shape: [batch_size, n_agents, n_actions]
    hidden state shape: [batch_size, n_agents, hidden_dim]
    """

    def __init__(self, input_dim, scheme, args):
        super().__init__()
        self.args = args
        self.obs_mean, self.obs_std = torch.zeros(scheme["obs"]["vshape"]).cuda(), torch.ones(scheme["obs"]["vshape"]).cuda()
        self.obs_count = 1e-4
        self.ext_count = 1e-4
        self.int_count = 1e-4
        self.padding_dim = input_dim - scheme["obs"]["vshape"]
        self.int_mean, self.int_std = torch.zeros(1).cuda(), torch.ones(1).cuda()
        self.ext_mean, self.ext_std = torch.zeros(1).cuda(), torch.ones(1).cuda()

        # local novelty target net
        # using for set an anchor for calculating the novelty of local observation
        # params frozen, don't update
        self.local_novelty_target = nn.Sequential(
            nn.Linear(input_dim, args.cur_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.cur_hidden_dim, args.cur_output_dim)
        )
        for params in self.local_novelty_target.parameters():
            params.requires_grad = False

        # local novelty predict net
        # using for calculating the novelty of local observation $r_i^{local} = f_i^local(o_i^t)$
        # this subnet backwards 2 loss(from global loss and local loss)
        self.local_novelty_predict = nn.Sequential(
            nn.Linear(input_dim, args.cur_hidden_dim),
            nn.Dropout(p=0.5),
            nn.Linear(args.cur_hidden_dim, args.cur_hidden_dim),
            nn.Dropout(p=0.5),
            nn.Linear(args.cur_hidden_dim, args.cur_output_dim)
        )

        # local Q net
        # using for calculating the local Q
        self.q_fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.q_rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, x, hidden):
        """
        x: [batch_size * n_agents, input_dim]
        hidden state: [batch_size, n_agents, hidden_dim]
        """
        x = F.relu(self.q_fc1(x))
        x = x.view(-1, x.size(-1))
#         print(x.shape)
        h_in = hidden.view(-1, self.args.rnn_hidden_dim)
#         print(h_in.shape)
        h_out = self.q_rnn(x, h_in)
#         print(h_out)
#         h_out = h_out.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)
        local_q = self.q_fc2(h_out).unsqueeze(0)
        return local_q, h_out

    def init_hidden(self):
        # trick, create hidden state on same device
        # batch size: 1
        return self.q_fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)
