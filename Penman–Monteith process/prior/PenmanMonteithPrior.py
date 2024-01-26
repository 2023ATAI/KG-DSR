import numpy as np

import configs.config
from prior.Prior import Prior


class PenmanMonteithPrior (Prior):
    """
    """

    def __init__(self, library, programs, targets, max, max_depth, scale):
        # Initialize the Prior class with the provided library and programs
        Prior.__init__(self, library, programs)
        # Is token of the library a terminal token : mask
        terminal_arity = 0
        self.mask_lib_is_terminal = (self.lib.get_choosable_prop("arity") == terminal_arity)
        self.mask_prob[:, :] = 1
        self.mask_prob_ = np.ones(self.mask_prob.shape)
        self.gaussian_vals = np.zeros(self.mask_prob.shape)
        self.max_depth = max_depth
        self.mask_depth_B = np.full(shape=(self.progs.batch_size), fill_value=False, dtype=bool)
        self.not_op_mask = np.ones(shape=self.lib.lib_choosable_name.shape)
        self.not_op_mask[np.where(self.lib.arity > 0)] = 0
        self.scale = float(scale)
        # _________________________________________________________________________________________________________
        # Define names for sections related to energy absorption, atmosphere absorption, and resistance
        X_ES_names = configs.config.X_ES_names
        X_AS_names = configs.config.X_AS_names
        X_RS_names = configs.config.X_RS_names

        # _________________________________________________________________________________________________________
        self.F_step = np.zeros((self.progs.batch_size, 4))
        # Create indices for tokens in each section
        # _________________________________________________________________________________________________________

        self.indices_ES = np.array(np.where(np.isin(self.lib.lib_choosable_name, X_ES_names)))
        self.indices_AS = np.array(np.where(np.isin(self.lib.lib_choosable_name, X_AS_names)))
        self.indices_RS = np.array((np.where(np.isin(self.lib.lib_choosable_name, X_RS_names))))
        self.index_list = []
        self.index_list.append(self.indices_ES)
        self.index_list.append(self.indices_AS)
        self.index_list.append(self.indices_RS)

        self.F1_mask = np.zeros(shape=self.lib.lib_choosable_name.shape)
        self.F1_mask[self.indices_ES] = 1
        self.F1_mask[np.where(self.lib.arity > 0)] = 1
        self.F2_mask = np.zeros(shape=self.lib.lib_choosable_name.shape)
        self.F2_mask[self.indices_AS] = 1
        self.F2_mask[np.where(self.lib.arity > 0)] = 1
        self.F3_mask = np.zeros(shape=self.lib.lib_choosable_name.shape)
        self.F3_mask[self.indices_RS] = 1
        self.F3_mask[np.where(self.lib.arity > 0)] = 1

        self.F12_mask = np.zeros(shape=self.lib.lib_choosable_name.shape)
        self.F12_mask[np.where(self.lib.arity == 2)] = 1

    def __call__(self):
        # Reset probs
        self.reset_mask_prob()
        # mask_depth = np.outer(np.where(self.progs.tokens.depth[:, self.progs.curr_step] > self.max_depth), np.where(self.progs.tokens.depth[:, self.progs.curr_step] !=self.lib.behavior_id[-1]))
        mask_depth = np.where((self.progs.tokens.depth[:, self.progs.curr_step] > self.max_depth) & (
                self.progs.tokens.depth[:, self.progs.curr_step] != self.progs.tokens.default_depth))
        self.mask_depth_B[:] = False
        self.mask_depth_B[mask_depth] = True
        name2idx = self.lib.lib_name_to_idx
        condition_matrix = np.zeros_like(self.F_step, dtype=bool)
        count_of_second = np.sum(self.progs.tokens.depth[:, :self.progs.curr_step + 1] == 2, axis=1)
        count_of_first = np.sum(self.progs.tokens.depth[:, :self.progs.curr_step + 1] == 1, axis=1)
        # Define operators for each part of the PM formula
        if self.progs.curr_step == 0:
            # At the first step, set the mask to 0 and enable the operator at index 3
            self.mask_prob[:, :] = 0
            self.mask_prob[:, 3] = 1
        elif self.progs.curr_step == 1:
            # At the second step, set the mask to 0 and enable the operator at index 1
            self.mask_prob[:, :] = 0
            self.mask_prob[:, 1] = 1
        else:
            # Initialize mask_prob to 0
            ## F1 的符号选择
            # Sample according to self.index_list[0]
            if (np.any(count_of_first == 1) and np.any(count_of_second == 1)):

                F1_index = (count_of_first == 1) & (count_of_second == 1)
                counts_F1 = np.equal.outer(self.progs.tokens.idx, self.index_list[0]).sum(axis=1)
                # ————————————————————————————————————————————————————————————————————————————————————————
                ### Forcing a maximum count for each section
                max_F1 = np.ones(self.index_list[0].shape[1])
                # Check if the count of tokens in the first section is less than the maximum allowed
                is_target_allowed_F1 = np.less(counts_F1, max_F1)
                # Set the mask for the PM formula to the energy absorption section
                self.mask_prob[np.where(F1_index), :] = self.F1_mask
                # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
                # # 期望的长度
                # If we want length = 3, Gaussian value must be maximum at step = 2 (i.e., when generating token number 3)
                length_loc = self.index_list[0].shape[1]
                # => step_loc = length_loc - 1
                self.step_loc = float(length_loc) - 1
                # Value of gaussian at all steps
                steps = np.arange(0, self.progs.max_time_step + 1)
                # gaussian_vals[step_loc] = gaussian_vals[steps[step_loc]]
                gaussian_vals__ = np.exp(-(steps - self.step_loc) ** 2 / (2 * self.scale*0.01))
                gaussian_vals__[int(self.step_loc)] = gaussian_vals__[int(self.step_loc)-1]
                gaussian_vals_ = np.tile(gaussian_vals__, (self.progs.batch_size, 1))
                # Before loc
                condition_matrix[F1_index, 0] = self.F_step[F1_index, 0] < self.step_loc
                rows_to_multiply_F1 = (F1_index) & (condition_matrix[:, 0])
                columns_to_multiply_F1 = self.mask_lib_is_terminal

                self.gaussian_vals[rows_to_multiply_F1, :] = np.tile(
                    gaussian_vals_[rows_to_multiply_F1, self.F_step[rows_to_multiply_F1, 0].astype(int)][:, np.newaxis],
                    (1, self.mask_prob.shape[1]))
                self.mask_prob[np.outer(rows_to_multiply_F1, columns_to_multiply_F1)] *= self.gaussian_vals[
                    np.outer(rows_to_multiply_F1, columns_to_multiply_F1)]
                # At loc: gaussian value = 1.
                # After loc
                # Identify the rows and columns based on conditions
                # ————————————————————————————————————————————————————————————————————————————————————————

                # Update the mask for the energy absorption section based on the maximum count allowed
                condition_matrix[F1_index, 0] = self.F_step[F1_index, 0] >= self.step_loc
                rows_to_multiply_F1 = (F1_index) & (condition_matrix[:, 0])
                columns_to_multiply_F1 = np.logical_not(self.mask_lib_is_terminal)
                self.gaussian_vals[rows_to_multiply_F1, :] = np.tile(
                    gaussian_vals_[rows_to_multiply_F1, self.F_step[rows_to_multiply_F1, 0].astype(int)][:, np.newaxis],
                    (1, self.mask_prob.shape[1]))
                # Perform element-wise multiplication for the specified rows and columns
                # Scale non-terminal tokens probs by gaussian
                self.mask_prob[np.outer(rows_to_multiply_F1, columns_to_multiply_F1)] *= self.gaussian_vals[
                    np.outer(rows_to_multiply_F1, columns_to_multiply_F1)]
                ######  判断可去掉
                # if np.any(rows_to_multiply_F1):
                #     self.mask_prob[np.outer(rows_to_multiply_F1, name2idx['SWdown'])] = 1
                #     self.mask_prob[np.outer(rows_to_multiply_F1, name2idx['LAI'])] = 1
                #     self.mask_prob[np.outer(rows_to_multiply_F1, name2idx['72'])] = 1

                self.mask_prob_[np.where(F1_index)[0][:, np.newaxis], self.index_list[0]] = is_target_allowed_F1[
                    np.where(F1_index)[0], 0].astype(float)
                self.mask_prob = self.mask_prob * self.mask_prob_

                self.F_step[F1_index, 0] += 1

                if np.sum(self.mask_depth_B) > 0:
                    self.mask_prob[self.mask_depth_B, :] = np.multiply(self.mask_prob[self.mask_depth_B, :],self.not_op_mask)
            # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
            ## F2 的符号选择
            if (np.any(count_of_first == 1) and np.any(count_of_second == 2)):
                F2_index = (count_of_first == 1) & (count_of_second == 2)
                counts_F2 = np.equal.outer(self.progs.tokens.idx, self.index_list[1]).sum(axis=1)
                ### Forcing a maximum count for each section
                max_F2 = np.ones(self.index_list[1].shape[1])
                # Check if the count of tokens in the second section is less than the maximum allowed
                is_target_allowed_F2 = np.less(counts_F2, max_F2)
                self.mask_prob[np.where((count_of_first == 1) & (count_of_second == 2)), :] = self.F2_mask

                # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
                # # 期望的长度
                # If we want length = 3, Gaussian value must be maximum at step = 2 (i.e., when generating token number 3)
                length_loc = self.index_list[1].shape[1]
                # => step_loc = length_loc - 1
                self.step_loc = float(length_loc) - 1
                # Value of gaussian at all steps
                steps = np.arange(0, self.progs.max_time_step + 1)
                # gaussian_vals[step_loc] = gaussian_vals[steps[step_loc]]
                gaussian_vals__ = np.exp(-(steps - self.step_loc) ** 2 / (2 * self.scale*0.01))
                gaussian_vals__[int(self.step_loc)] = gaussian_vals__[int(self.step_loc)-1]
                gaussian_vals_ = np.tile(gaussian_vals__, (self.progs.batch_size, 1))
                # Before loc
                condition_matrix[F2_index, 1] = self.F_step[F2_index, 1] < self.step_loc
                rows_to_multiply_F2 = (F2_index) & (condition_matrix[:, 1])
                columns_to_multiply_F2 = self.mask_lib_is_terminal
                self.gaussian_vals[rows_to_multiply_F2, :] = np.tile(
                    gaussian_vals_[rows_to_multiply_F2, self.F_step[rows_to_multiply_F2, 1].astype(int)][:, np.newaxis],
                    (1, self.mask_prob.shape[1]))
                self.mask_prob[np.outer(rows_to_multiply_F2, columns_to_multiply_F2)] *= self.gaussian_vals[
                    np.outer(rows_to_multiply_F2, columns_to_multiply_F2)]
                # At loc: gaussian value = 1.
                # After loc
                # Identify the rows and columns based on conditions
                condition_matrix[F2_index, 1] = self.F_step[F2_index, 1] >= self.step_loc
                rows_to_multiply_F2 = (F2_index) & (condition_matrix[:, 1])
                columns_to_multiply_F2 = np.logical_not(self.mask_lib_is_terminal)
                self.gaussian_vals[rows_to_multiply_F2, :] = np.tile(
                    gaussian_vals_[rows_to_multiply_F2, self.F_step[rows_to_multiply_F2, 1].astype(int)][:,
                    np.newaxis], (1, self.mask_prob.shape[1]))
                # Perform element-wise multiplication for the specified rows and columns
                # Scale non-terminal tokens probs by gaussian
                self.mask_prob[np.outer(rows_to_multiply_F2, columns_to_multiply_F2)] *= self.gaussian_vals[
                    np.outer(rows_to_multiply_F2, columns_to_multiply_F2)]
                # if np.any(rows_to_multiply_F2):
                #     self.mask_prob[rows_to_multiply_F2, name2idx['WP']] = 1
                #     self.mask_prob[rows_to_multiply_F2, name2idx['FC']] = 1
                #     self.mask_prob[rows_to_multiply_F2, name2idx['swc_root']] = 1
                self.mask_prob_[np.where(F2_index)[0][:, np.newaxis], self.index_list[1]] = is_target_allowed_F2[
                    np.where(F2_index)[0], 0].astype(float)
                self.mask_prob = self.mask_prob * self.mask_prob_

                self.F_step[F2_index, 1] += 1
                if np.sum(self.mask_depth_B) > 0:
                    self.mask_prob[self.mask_depth_B, :] = np.multiply(self.mask_prob[self.mask_depth_B, :],
                                                                       self.not_op_mask)
            # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # ## F3 的符号选择
            if (np.any(count_of_first == 2)):
                F3_index = ((count_of_first == 2) )
                counts_F3 = np.equal.outer(self.progs.tokens.idx, self.index_list[2]).sum(axis=1)
                ### Forcing a maximum count for each section
                max_F3 = np.ones(self.index_list[2].shape[1])
                max_F3[-2:] = 2
                # Check if the count of tokens in the second section is less than the maximum allowed
                is_target_allowed_F3 = np.less(counts_F3, max_F3)
                self.mask_prob[np.where(F3_index), :] = self.F3_mask

                # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
                # # 期望的长度
                # If we want length = 3, Gaussian value must be maximum at step = 2 (i.e., when generating token number 3)
                length_loc = self.index_list[2].shape[1]
                # => step_loc = length_loc - 1
                self.step_loc = float(length_loc) - 1
                # Value of gaussian at all steps
                steps = np.arange(0, self.progs.max_time_step + 1)
                # gaussian_vals[step_loc] = gaussian_vals[steps[step_loc]]
                gaussian_vals__ = np.exp(-(steps - self.step_loc) ** 2 / (2 * self.scale))
                #gaussian_vals__[int(self.step_loc)] = gaussian_vals__[int(self.step_loc)-1]
                gaussian_vals_ = np.tile(gaussian_vals__, (self.progs.batch_size, 1))

                # Before loc
                condition_matrix[F3_index, 2] = self.F_step[F3_index, 2] < self.step_loc
                rows_to_multiply_F3 = (F3_index) & (condition_matrix[:, 2])
                columns_to_multiply_F3 = self.mask_lib_is_terminal
                self.gaussian_vals[rows_to_multiply_F3, :] = np.tile(
                    gaussian_vals_[rows_to_multiply_F3, self.F_step[rows_to_multiply_F3, 2].astype(int)][:,
                    np.newaxis],
                    (1, self.mask_prob.shape[1]))
                self.mask_prob[np.outer(rows_to_multiply_F3, columns_to_multiply_F3)] *= self.gaussian_vals[
                    np.outer(rows_to_multiply_F3, columns_to_multiply_F3)]
                # At loc: gaussian value = 1.
                # After loc
                # Identify the rows and columns based on conditions
                condition_matrix[F3_index, 2] = self.F_step[F3_index, 2] >= self.step_loc
                rows_to_multiply_F3 = (F3_index) & (condition_matrix[:, 2])
                columns_to_multiply_F3 = np.logical_not(self.mask_lib_is_terminal)
                self.gaussian_vals[rows_to_multiply_F3, :] = np.tile(
                    gaussian_vals_[rows_to_multiply_F3, self.F_step[rows_to_multiply_F3, 2].astype(int)][:,
                    np.newaxis], (1, self.mask_prob.shape[1]))
                # Perform element-wise multiplication for the specified rows and columns
                # Scale non-terminal tokens probs by gaussian
                self.mask_prob[np.outer(rows_to_multiply_F3, columns_to_multiply_F3)] *= self.gaussian_vals[
                    np.outer(rows_to_multiply_F3, columns_to_multiply_F3)]
                # if np.any(rows_to_multiply_F3):
                #     self.mask_prob[rows_to_multiply_F3, name2idx['VPD']] = 1
                #     self.mask_prob[rows_to_multiply_F3, name2idx['0.1914']] = 1
                self.mask_prob_[np.where(F3_index)[0][:, np.newaxis], self.index_list[2]] = is_target_allowed_F3[
                    np.where(F3_index)[0], 0].astype(float)
                self.mask_prob = self.mask_prob * self.mask_prob_

                self.F_step[F3_index, 2] += 1

                if np.sum(self.mask_depth_B) > 0:
                    self.mask_prob[self.mask_depth_B, :] = np.multiply(self.mask_prob[self.mask_depth_B, :],
                                                                       self.not_op_mask)
        return self.mask_prob

    def __repr__(self):
        return "PenmanMonteithPrior"
#
