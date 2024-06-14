# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np


class UIPS(object):
    def __init__(self, user_count, item_count, cate_count, cate_list, para, random_seed = 1234,hidden_units=128):
        tf.set_random_seed(random_seed)
        self.user_count = user_count
        self.item_count = item_count
        self.cate_count = cate_count
        self.cate_list = tf.convert_to_tensor(cate_list, dtype=tf.float32)  # [item_count,cate_count]
        self.hidden_units = hidden_units
        self.lambda_ = tf.cast(para['lambda_'], tf.float32)
        self.eta_ = tf.cast(para['eta_'], tf.float32)
        self.para = para
        print('[Model_UIPS] para: ', para)

        ##################PLACEHOLDER#######################
        self.A = tf.Variable(initial_value=np.identity(hidden_units), trainable=False, dtype=tf.float32)
        self.u = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.display = tf.placeholder(tf.float32, [None, ])  # [B]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        self.initA = tf.assign(self.A, np.identity(hidden_units))

        #########################PI Network ##############################
        self.item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
        self.item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))
        self.cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
        self.user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        self.u_dense_layer = tf.layers.Dense(hidden_units, name='u_emb_mlp')
        self.dense_layer = tf.layers.Dense(1, use_bias=False, name='pi_final_mlp')

        self.u_emb = self.getUemb()  # user_state embedding
        self.i_emb, self.i_b = self.getIemb(self.item_emb_w, self.cate_emb_w, self.item_b)  # [B,H], [B,]

        ###################### BETA Network ###########################
        self.beta_sg_u_emb = tf.identity(tf.stop_gradient(self.u_emb))
        self.loss_beta, self.beta_ips_prob, self.beta_uncertainty, self.beta_updateA = self.build_beta_network()

        ####################### PI Network ##########################
        # pi network only consider samples with display=1
        displayed_index = tf.where(self.display > 0)
        dis_u_emb = tf.squeeze(tf.gather(self.u_emb, displayed_index), axis=1)  # [disB, d]
        dis_i_emb = tf.squeeze(tf.gather(self.i_emb, displayed_index), axis=1)
        dis_i_b = tf.squeeze(tf.gather(self.i_b, displayed_index), axis=1)  # [disB, ]
        dis_size = tf.shape(dis_u_emb)[0]
        dis_n_u_emb = tf.tile(tf.expand_dims(dis_u_emb, 1), [1, dis_size, 1])  # [disB, disB, d]
        dis_n_i_emb = tf.tile(tf.expand_dims(dis_i_emb, 0), [dis_size, 1, 1])
        u_emb_i_emb = tf.multiply(dis_n_u_emb, dis_n_i_emb)  # [disB, disB, d]
        pi_score = tf.squeeze(self.dense_layer.apply(u_emb_i_emb), axis=2) + dis_i_b
        self.pi_prob = tf.diag_part(tf.keras.layers.Softmax(axis=-1)(pi_score))

        ############################ LOSS #############################
        self.importance_weight = self.getIPS(displayed_index)
        onehot_labels = tf.eye(tf.shape(dis_u_emb)[0], tf.shape(dis_u_emb)[0])
        loss_per_example = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=pi_score)
        dis_y = tf.squeeze(tf.gather(self.y, displayed_index), axis=1)
        temp_mul = tf.multiply(dis_y - self.para['lambdaDiff'], loss_per_example)
        self.loss_pi = tf.reduce_mean(tf.multiply(self.importance_weight, temp_mul))
        tf.summary.scalar('loss/loss_per_example: ', tf.reduce_mean(temp_mul))
        tf.summary.scalar('loss/loss_pi: ', tf.reduce_mean(self.loss_pi))

        ############# UPDATE GRADIENT ############

        self.loss = self.loss_beta + self.loss_pi
        tf.summary.scalar('loss/loss: ', tf.reduce_mean(self.loss))

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        ### joint train
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)
        ### train beta
        gradients_beta = tf.gradients(self.loss_beta, trainable_params)
        clip_gradients_beta, _ = tf.clip_by_global_norm(gradients_beta, 5)
        self.train_op_beta = self.opt.apply_gradients(zip(clip_gradients_beta, trainable_params),
                                                      global_step=self.global_step)
        ### train pi
        gradients_pi = tf.gradients(self.loss_pi, trainable_params)
        clip_gradients_pi, _ = tf.clip_by_global_norm(gradients_pi, 5)
        self.train_op_pi = self.opt.apply_gradients(zip(clip_gradients_pi, trainable_params),
                                                    global_step=self.global_step)

        self.merged = tf.summary.merge_all()
        ############# USED TO EVALUATE############
        self.eval_all_i = tf.range(start=0, limit=self.item_count, delta=1)
        self.eval_item_emb, self.eval_item_b = self.getAllItem()
        eval_u_size = tf.shape(self.u_emb)[0]
        eval_i_size = tf.shape(self.eval_item_emb)[0]
        n_u_emb_place = tf.expand_dims(self.u_emb, 1)
        n_u_emb_place = tf.tile(n_u_emb_place, [1, eval_i_size, 1])
        n_iemb_place = tf.expand_dims(self.eval_item_emb, 0)
        n_iemb_place = tf.tile(n_iemb_place, [eval_u_size, 1, 1])
        u_emb_i_emb_place = tf.multiply(n_u_emb_place, n_iemb_place)
        self.eval_score = self.dense_layer.apply(u_emb_i_emb_place)
        self.eval_score = tf.squeeze(self.eval_score, axis=2) + self.eval_item_b

        # output
        self.dis_i = tf.squeeze(tf.gather(self.i, displayed_index), axis=1)

    def build_beta_network(self):
        ############# Beta Network ############
        # beta network
        self.beta_item_emb_w = tf.get_variable("beta_item_emb_w", [self.item_count, self.hidden_units // 2])
        self.beta_item_b = tf.get_variable("beta_item_b", [self.item_count], initializer=tf.constant_initializer(0.0))
        self.beta_cate_emb_w = tf.get_variable("beta_cate_emb_w", [self.cate_count, self.hidden_units // 2])
        self.beta_dense_layer = tf.layers.Dense(1, use_bias=False, name='beta_final_mlp')
        self.beta_i_emb, self.beta_i_b = self.getIemb(self.beta_item_emb_w, self.beta_cate_emb_w, self.beta_item_b)

        ##################### Beta Network Loss #####################
        # get loss
        batch_size = tf.shape(self.beta_sg_u_emb)[0]
        n_u_emb = tf.expand_dims(self.beta_sg_u_emb, 1)
        n_u_emb = tf.tile(n_u_emb, [1, batch_size, 1])
        beta_n_iemb = tf.expand_dims(self.beta_i_emb, 0)
        beta_n_iemb = tf.tile(beta_n_iemb, [batch_size, 1, 1])
        beta_u_emb_i_emb = tf.multiply(tf.stop_gradient(n_u_emb), beta_n_iemb)
        self.logits_beta = self.beta_dense_layer.apply(beta_u_emb_i_emb)
        self.logits_beta = tf.squeeze(self.logits_beta, axis=2) + self.beta_i_b

        onehot_labels = tf.eye(tf.shape(self.u_emb)[0], tf.shape(self.u_emb)[0])
        loss_per_example_beta = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
                                                                           logits=self.logits_beta)
        self.loss_beta = tf.reduce_mean(tf.multiply(self.display, loss_per_example_beta))
        tf.summary.scalar('loss/loss_beta: ', tf.reduce_mean(self.loss_beta))

        # get beta prob in IPS
        displayed_index = tf.where(self.display > 0)
        dis_beta_u_emb = tf.squeeze(tf.gather(self.beta_sg_u_emb, displayed_index), axis=1)
        dis_beta_i_emb = tf.squeeze(tf.gather(self.beta_i_emb, displayed_index), axis=1)
        dis_beta_i_b = tf.squeeze(tf.gather(self.beta_i_b, displayed_index), axis=1)  # [disB, ]
        dis_size = tf.shape(dis_beta_u_emb)[0]
        dis_beta_n_u_emb = tf.tile(tf.expand_dims(dis_beta_u_emb, 1), [1, dis_size, 1])  # [disB, disB, d]
        dis_beta_n_i_emb = tf.tile(tf.expand_dims(dis_beta_i_emb, 0), [dis_size, 1, 1])
        dis_beta_u_emb_i_emb = tf.multiply(dis_beta_n_u_emb, dis_beta_n_i_emb)  # [disB, disB, d]

        self.logits_beta_dis = self.beta_dense_layer.apply(dis_beta_u_emb_i_emb)
        self.logits_beta_dis = tf.squeeze(self.logits_beta_dis, axis=2) + dis_beta_i_b
        beta_prob = tf.keras.layers.Softmax(axis=-1)(self.logits_beta_dis)
        epsilon = 0.0000001
        self.beta_ips_prob = tf.stop_gradient(tf.diag_part(beta_prob)) + epsilon
        ################### Uncertainty #############################
        self.beta_uncertainty, self.beta_updateA = self.calUncertainty(dis_beta_u_emb_i_emb)

        return self.loss_beta, self.beta_ips_prob, self.beta_uncertainty, self.beta_updateA

    def calUncertainty(self, dis_beta_u_emb_i_emb):
        dis_size = tf.shape(dis_beta_u_emb_i_emb)[0]
        grad_beta_2D = dis_beta_u_emb_i_emb  # [disB,disB,128]
        temp_G = tf.reshape(grad_beta_2D, [dis_size * dis_size, self.hidden_units])
        ind = tf.tile(tf.reshape(tf.eye(dis_size, dis_size), [dis_size * dis_size, -1]), [1, self.hidden_units])
        grad_beta_diag = tf.boolean_mask(tensor=temp_G, mask=ind)  # [disB,128]
        grad_beta_diag = tf.reshape(grad_beta_diag, [-1, self.hidden_units])  # [disB, 128]--diagnoal veral
        grad_beta_diag, _ = tf.linalg.normalize(grad_beta_diag, axis=-1)
        invA = tf.linalg.inv(self.A)
        tf.summary.scalar('A/invA_norm: ', tf.norm(invA))
        tf.summary.scalar('A/A_norm:', tf.norm(self.A))

        uncertainty = tf.diag_part(
            tf.matmul(tf.matmul(grad_beta_diag, invA), tf.transpose(grad_beta_diag)) / self.hidden_units)
        uncertainty = tf.math.sqrt(uncertainty)  # [disB,1]
        additive = tf.matmul(tf.transpose(grad_beta_diag), grad_beta_diag) / self.hidden_units
        tf.summary.scalar('A/additive:', tf.norm(additive))

        self.beta_updateA_op = tf.assign(self.A, tf.add(self.A, additive))
        return tf.stop_gradient(uncertainty), self.beta_updateA_op

    def getAllItem(self):
        i = tf.range(start=0, limit=self.item_count, delta=1)
        all_i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(self.item_emb_w, i),  # [B,h/2]
            tf.linalg.matmul(self.cate_list, self.cate_emb_w)
        ], axis=-1)
        all_i_b = self.item_b
        return all_i_emb, all_i_b

    def getIPS(self, displayed_index):
        pi_prob_sg = tf.stop_gradient(self.pi_prob)
        ipsweight = tf.math.divide(pi_prob_sg, self.beta_ips_prob)
        tf.summary.scalar('ipsweight/ipsweight_mean', tf.reduce_mean(ipsweight))
        tf.summary.scalar('ipsweight/ipsweight_min', tf.reduce_min(ipsweight))
        tf.summary.scalar('ipsweight/ipsweight_max', tf.reduce_max(ipsweight))

        ##########################UNCERTAINTY REWEIGHT######################
        diag_beta_dis_uty = tf.exp(-self.para['gamma'] * self.beta_uncertainty)
        tf.summary.scalar('beta_dis_uty/min_beta_dis_uty', tf.reduce_min(self.beta_uncertainty))
        tf.summary.scalar('beta_dis_uty/max_beta_dis_uty', tf.reduce_max(self.beta_uncertainty))
        tf.summary.scalar('beta_dis_uty/mean_beta_dis_uty', tf.reduce_mean(self.beta_uncertainty))
        tf.summary.scalar('beta_dis_uty/min_exp_beta_dis_uty', tf.reduce_min(diag_beta_dis_uty))
        tf.summary.scalar('beta_dis_uty/max_exp_beta_dis_uty', tf.reduce_max(diag_beta_dis_uty))
        tf.summary.scalar('beta_dis_uty/mean_exp_beta_dis_uty', tf.reduce_mean(diag_beta_dis_uty))

        pi_beta_U = tf.math.divide(tf.square(pi_prob_sg),
                                   tf.multiply(tf.square(self.beta_ips_prob), diag_beta_dis_uty) + 0.0000001)
        tf.summary.scalar('pi_beta_U/min_pi_beta_U', tf.reduce_min(pi_beta_U))
        tf.summary.scalar('pi_beta_U/max_pi_beta_U', tf.reduce_max(pi_beta_U))
        tf.summary.scalar('pi_beta_U/mean_pi_beta_U', tf.reduce_mean(pi_beta_U))

        phi_sa = self.lambda_ / (self.lambda_ / self.eta_ * diag_beta_dis_uty + pi_beta_U + 0.000001)
        tf.summary.scalar('phi_sa/ori_phi_sa_mean', tf.reduce_mean(phi_sa))
        tf.summary.scalar('phi_sa/ori_phi_sa_min', tf.reduce_min(phi_sa))
        tf.summary.scalar('phi_sa/ori_phi_sa_max', tf.reduce_max(phi_sa))

        if self.para['normalize_phi_sa'] > 0:
            phi_sa = phi_sa / (tf.reduce_sum(phi_sa) + 0.000001)
        tf.summary.scalar('phi_sa/phi_sa_mean', tf.reduce_mean(phi_sa))
        tf.summary.scalar('phi_sa/phi_sa_min', tf.reduce_min(phi_sa))
        tf.summary.scalar('phi_sa/phi_sa_max', tf.reduce_max(phi_sa))

        #################################GET Final IPS #######################
        is_capping = tf.cast(tf.equal(self.global_epoch_step, 0), dtype=tf.float32) * self.para['cappingFirstEpoch']
        is_capping_vec = tf.cast(is_capping * tf.ones_like(ipsweight), tf.bool)
        ips_final_capping = tf.where(tf.greater(ipsweight, self.para['cappingThre']),
                                     self.para['cappingThre'] * tf.ones_like(ipsweight), ipsweight)
        ips_final_un = tf.multiply(ipsweight, phi_sa)

        ips_final = tf.where(is_capping_vec, ips_final_capping, ips_final_un)

        # print_op = tf.print('ips_final_un, ', tf.shape(ips_final_un), 'ips_final_capping: ', tf.shape(ips_final_capping), 'ips_final: ', tf.shape(ips_final))
        # with tf.control_dependencies([print_op]):
        #   ips_final = tf.identity(ips_final)

        tf.summary.scalar('ips_final/is_capping_vec', tf.reduce_sum(tf.cast(is_capping_vec, tf.float32)))
        tf.summary.scalar('ips_final/ips_final_capping', tf.reduce_mean(ips_final_capping))
        tf.summary.scalar('ips_final/ips_final_un', tf.reduce_mean(ips_final_un))
        tf.summary.scalar('ips_final/ips_final_mean', tf.reduce_mean(ips_final))
        tf.summary.scalar('ips_final/ips_final_min', tf.reduce_min(ips_final))
        tf.summary.scalar('ips_final/ips_final_max', tf.reduce_max(ips_final))

        ips_final_cond = tf.equal(ips_final, tf.reduce_max(ips_final))
        tf.summary.scalar('debug/ips_final_max_phi_sa: ',
                          tf.reduce_mean(tf.boolean_mask(tensor=phi_sa, mask=ips_final_cond)))
        tf.summary.scalar('debug/ips_final_max_ipsweight: ',
                          tf.reduce_mean(tf.boolean_mask(tensor=ipsweight, mask=ips_final_cond)))
        tf.summary.scalar('debug/ips_final_max_diag_beta_dis_uty: ',
                          tf.reduce_mean(tf.boolean_mask(tensor=diag_beta_dis_uty, mask=ips_final_cond)))
        tf.summary.scalar('debug/ips_final_max_pi_beta_U: ',
                          tf.reduce_mean(tf.boolean_mask(tensor=pi_beta_U, mask=ips_final_cond)))
        tf.summary.scalar('debug/ips_final_max_ips_final: ',
                          tf.reduce_mean(tf.boolean_mask(tensor=tf.multiply(ipsweight, phi_sa), mask=ips_final_cond)))

        return tf.stop_gradient(ips_final)

    def getIemb(self, item_emb_w, cate_emb_w, item_b):
        ic = tf.gather(self.cate_list, self.i)
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),  # [B,h/2]
            tf.linalg.matmul(ic, cate_emb_w),  # [B,cate_count],[cate_count,h/2]
        ], axis=1)
        i_b = tf.gather(item_b, self.i)
        return i_emb, i_b

    def getUemb(self):
        u_emb_0 = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        hc = tf.gather(self.cate_list, self.hist_i)  # self.hist_i [B,T,cate_count]
        h_emb = tf.concat([
            tf.nn.embedding_lookup(self.item_emb_w, self.hist_i),  # [B,T,H/2]
            tf.linalg.matmul(hc, self.cate_emb_w),  # [B,T,cate_count] [cate_count,h/2]  # 会出现nan值
        ], axis=2)
        mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32)  # [disB, T]
        mask = tf.expand_dims(mask, -1)  # [B, T, 1]
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B, T, H]
        h_emb *= mask  # [B, T, H]
        hist = h_emb
        hist = tf.reduce_sum(hist, 1)
        hist = tf.math.divide_no_nan(hist,
                                     tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, self.hidden_units]), tf.float32))
        hist = tf.reshape(hist, [-1, self.hidden_units])
        hist = tf.layers.dense(hist, self.hidden_units)
        u_emb = tf.concat([u_emb_0, hist], axis=-1)  # [B,2H]
        u_emb = tf.layers.batch_normalization(inputs=u_emb)
        u_emb = self.u_dense_layer.apply(u_emb)  # [B,H]
        return u_emb

    def train_pi(self, sess, uij, l):
        loss, _,dis_i,beta_prob,beta_uncertainty = sess.run([self.loss_pi, self.train_op_pi,self.dis_i,self.beta_ips_prob, self.beta_uncertainty], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l,
            self.display: uij[5]
        })
       
        return loss

    def train_beta(self, sess, uij, l):
        loss, _, _ = sess.run([self.loss_beta, self.train_op_beta, self.beta_updateA],
                                           feed_dict={
                                               self.u: uij[0],
                                               self.i: uij[1],
                                               self.y: uij[2],
                                               self.hist_i: uij[3],
                                               self.sl: uij[4],
                                               self.lr: l,
                                               self.display: uij[5]
                                           })
        return loss

    def train(self, sess, uij, l):
        loss, _, _ = sess.run([self.loss, self.train_op, self.beta_updateA], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l,
            self.display: uij[5],
        })
        return loss

    def run_evaluate_user(self, sess, u, u_hist, sl):
        eval_score = sess.run([self.eval_score], feed_dict={
            self.u: u,
            self.hist_i: u_hist,
            self.sl: sl
        })
        return eval_score

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

