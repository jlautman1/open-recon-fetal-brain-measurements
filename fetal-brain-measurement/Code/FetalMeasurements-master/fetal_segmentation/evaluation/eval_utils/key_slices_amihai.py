class KeySlices:
    def __init__(self, data_path, workbook):
        self.workbook = workbook
        self.data_path = data_path
        self.pat_slice_inds = {}
        self.pat_slice_info = {}
        self.new_tumors_with_bl = {}
        self.format = workbook.add_format({'bold': True, 'font_size': 20})

    def write_images(self):
        self.get_slices_info()  # populate info about tumors
        self.get_slices()  # find key slice for each baseline tumor
        self.process_patients()

    def add_image(self, worksheet, image_path, image_num, image_ind):
        row = 0 + 27 * image_num
        worksheet.write(row, 0, "Slice index:{}".format(image_ind), self.format)
        worksheet.insert_image(row + 1, 0, image_path)
        return row + 27

    def process_patients(self):
        for pat in sorted(os.listdir(self.data_path)):
            worksheet = self.workbook.add_worksheet("Key slices- {}".format(pat))

            bl_seg_path = self.data_path + pat + '/nii/bl_combine_seg.nii'
            fu_seg_path = self.data_path + pat + '/nii/fu_combine_seg.nii'
            bl_scan_path = self.data_path + pat + '/nii/0{}_bl.nii'.format(pat)
            fu_scan_path = self.data_path + pat + '/nii/reg_0{}_fu_01.nii'.format(pat)
            _, bl_seg = load_case(bl_seg_path)
            _, fu_seg = load_case(fu_seg_path)
            _, bl_scan = load_case(bl_scan_path)
            _, fu_scan = load_case(fu_scan_path)

            # slice_img = self.process_slice(bl_seg[:, :, 150], fu_seg[:, :, 150], bl_scan[:, :, 150], fu_scan[:, :, 150])
            if not os.path.exists("images/{}/".format(pat)):
                os.mkdir("images/{}/".format(pat))
            for i, slice_ind in enumerate(self.pat_slice_inds[pat]):
                bl_seg_slice = bl_seg[:, :, slice_ind]
                fu_seg_slice = fu_seg[:, :, slice_ind]
                bl_scan_slice = bl_scan[:, :, slice_ind]
                fu_scan_slice = fu_scan[:, :, slice_ind]
                slice_img = self.process_slice(bl_seg_slice, fu_seg_slice, bl_scan_slice, fu_scan_slice)
                plt.imsave("images/{0}/ks-{1}.png".format(pat, i), slice_img.astype(np.uint8))

                img_path = "images/{0}/ks-{1}.png".format(pat, i)
                end_row = self.add_image(worksheet, img_path, i, slice_ind)

            self.write_new_tumors(worksheet, pat, end_row)

    def process_slice(self, bl_seg_slice, fu_seg_slice, bl_scan_slice, fu_scan_slice):
        fu_seg_delta, fu_seg_new = np.zeros((fu_seg_slice.shape)), np.zeros((fu_seg_slice.shape))
        # split new and delta
        fu_seg_delta[fu_seg_slice == 1] = 1
        fu_seg_new[fu_seg_slice == 2] = 1

        # get edges of marks
        bl_seg_edge = canny(bl_seg_slice, sigma=0.1, use_quantiles=True, high_threshold=0.9, low_threshold=0.2)
        fu_delta_edge = canny(fu_seg_delta, sigma=0.1, use_quantiles=True, high_threshold=0.9, low_threshold=0.2)
        fu_new_edge = canny(fu_seg_new, sigma=0.1, use_quantiles=True, high_threshold=0.9, low_threshold=0.2)

        bl_scan_slice = rescale_intensity(bl_scan_slice, out_range=(0, 255))
        fu_scan_slice = rescale_intensity(fu_scan_slice, out_range=(0, 255))

        bl_rgb_slice = np.dstack([bl_scan_slice, bl_scan_slice, bl_scan_slice])
        bl_rgb_slice[:, :, 0] += MARK_INTENSITY * bl_seg_edge

        fu_rgb_slice = np.dstack([fu_scan_slice, fu_scan_slice, fu_scan_slice])
        fu_rgb_slice[:, :, 0] += MARK_INTENSITY * fu_delta_edge
        fu_rgb_slice[:, :, 1] += MARK_INTENSITY * fu_new_edge

        final_img = np.hstack([bl_rgb_slice, fu_rgb_slice])

        # show_slice(final_img)
        return final_img

    def get_slices_info(self):
        for pat in sorted(os.listdir(self.data_path)):
            self.pat_slice_info[pat] = {}
            pat_path = self.data_path + pat + '/nii'
            for file in os.listdir(pat_path):
                if 'seg' in file and 'combine' not in file:
                    seg_path = pat_path + '/' + file
                    _, seg = load_case(seg_path)
                    seg_inds = np.where(seg == 1)[2]
                    ind_min = np.min(seg_inds)
                    ind_max = np.max(seg_inds)
                    nonzero_per_slice = np.sum(np.count_nonzero(seg, axis=1), axis=1)  # TODO doesnt work
                    quant25 = np.quantile(nonzero_per_slice, q=0.25)
                    quant75 = np.quantile(nonzero_per_slice, q=0.75)

                    if 'new' in file:
                        key = file
                        self.pat_slice_info[pat][key] = [ind_min, ind_max, quant25, quant75]

                    elif 'BL' in file:
                        key = file
                        self.pat_slice_info[pat][key] = [ind_min, ind_max, quant25, quant75]

                    elif 'delta' in file:
                        key = file
                        self.pat_slice_info[pat][key] = [ind_min, ind_max, quant25, quant75]

    def find_bl_slice(self, pat, tumor_info):
        ind_min, ind_max, quant25, quant75 = tumor_info
        best_slice = [0, []]  # [bl_ind, number of new]
        for slice_ind in range(ind_min, ind_max + 1):
            chosen_new_names = []
            for tumor in self.pat_slice_info[pat]:
                if 'new' in tumor:
                    new_min, new_max = self.pat_slice_info[pat][tumor][0], self.pat_slice_info[pat][tumor][1]
                    if slice_ind in range(new_min, new_max + 1):
                        chosen_new_names.append(tumor)

            if len(chosen_new_names) >= len(best_slice[1]):
                best_slice[0] = slice_ind
                best_slice[1] = chosen_new_names

        return best_slice

    def get_slices(self):
        for pat in self.pat_slice_info:
            self.pat_slice_inds[pat] = []
            self.new_tumors_with_bl[pat] = []
            for tumor in self.pat_slice_info[pat]:
                if 'BL' in tumor:
                    slice_ind, new_tumor_names = self.find_bl_slice(pat, self.pat_slice_info[pat][tumor])
                    self.pat_slice_inds[pat].append(slice_ind)
                    self.new_tumors_with_bl[pat] += new_tumor_names

    def process_new_slices(self, fu_slices, seg_slices, tumor_inds):
        processed_slices = []
        final_images = {}
        last_inds = []
        for i, tumor in enumerate(fu_slices):
            slice = fu_slices[tumor]
            slice = rescale_intensity(slice, out_range=(0, 255))
            rgb_slice = np.dstack([slice, slice, slice])

            seg_edge = canny(seg_slices[tumor], sigma=0.01, use_quantiles=True, high_threshold=0.9, low_threshold=0.2)
            rgb_slice[:, :, 1] += MARK_INTENSITY * seg_edge
            processed_slices.append(rgb_slice)
            last_inds.append(tumor_inds[tumor])

            if len(last_inds) % 3 == 0:
                stack_img = np.hstack([processed_slices[i - 2], processed_slices[i - 1], processed_slices[i]])
                final_images[tuple(last_inds)] = stack_img
                last_inds.clear()

        if len(processed_slices) % 3 == 1:
            final_images[tuple(last_inds)] = processed_slices[-1]
        elif len(processed_slices) % 3 == 2:
            stack = np.hstack([processed_slices[-1], processed_slices[-2]])
            final_images[tuple(last_inds)] = stack

        return final_images

    def write_new_tumors(self, worksheet, pat, start_row):
        worksheet.write(start_row, 0, "New tumors only", self.format)
        start_row += 1
        tumors_left = set(self.pat_slice_info[pat].keys()) - set(self.new_tumors_with_bl[pat])
        new_tumors_left = [tumor for tumor in tumors_left if 'new' in tumor]
        if len(new_tumors_left) > 0:
            new_tumor_seg_paths = {new_tumor: self.data_path + pat + '/nii/' + new_tumor for new_tumor in
                                   new_tumors_left}
            new_tumor_inds = {
                new_tumor: (self.pat_slice_info[pat][new_tumor][0] + self.pat_slice_info[pat][new_tumor][1]) // 2
                for new_tumor in new_tumors_left}  # get middle slice ind for tumor

            _, fu = load_case(self.data_path + pat + '/nii/reg_0{}_fu_01.nii'.format(pat))
            new_tumor_slices = {new_tumor: fu[:, :, new_tumor_inds[new_tumor]] for new_tumor in new_tumor_inds}
            new_tumor_seg_slices = {new_tumor: load_case(new_tumor_seg_paths[new_tumor])[1] for new_tumor in
                                    new_tumor_seg_paths}
            new_tumor_seg_slices = {new_tumor: new_tumor_seg_slices[new_tumor][:, :, new_tumor_inds[new_tumor]] for
                                    new_tumor in
                                    new_tumor_seg_slices}

            final_images = self.process_new_slices(new_tumor_slices, new_tumor_seg_slices, new_tumor_inds)
            for i, inds in enumerate(final_images):
                plt.imsave("images/{0}/new-{1}.png".format(pat, i), final_images[inds].astype(np.uint8))
                row = start_row + i * 27
                extra = ["" for i in range(3 - len(inds))]
                worksheet.write(row, 0, 'Slice indicies: {0}, {1}, {2}'.format(*inds, *extra), self.format)
                worksheet.insert_image(row + 1, 0, "images/{0}/new-{1}.png".format(pat, i))