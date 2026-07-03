# Optional: vendor/TruFor/TruFor_train_test/dataset/data_core.py 패치
#
# train_trufor_video_forgery.py 는 런타임 monkey-patch 로 FSVIDEO 를 등록하므로
# 이 패치는 **필수가 아닙니다**. vendor 를 직접 수정해도 될 때만 적용하세요.
#
# 1) import 추가:
#    from dataset.dataset_ForenShieldVideo import ForenShieldVideo
#
# 2) mode == "train" 블록에 추가:
#        if 'FSVIDEO' in training_set:
#            fs = config.FORENSHIELD
#            self.dataset_list.append(
#                ForenShieldVideo(
#                    crop_size, grid_crop,
#                    cache_root=fs.CACHE_ROOT,
#                    list_file=fs.TRAIN_LIST,
#                    aug=aug,
#                )
#            )
#
# 3) mode == "valid" 블록에 추가:
#        if 'FSVIDEO' in valid_set:
#            fs = config.FORENSHIELD
#            self.dataset_list.append(
#                ForenShieldVideo(
#                    crop_size, grid_crop,
#                    cache_root=fs.CACHE_ROOT,
#                    list_file=fs.VALID_LIST,
#                    max_dim=max_dim,
#                    aug=aug,
#                )
#            )
