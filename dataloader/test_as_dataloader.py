import torch
from torch.utils.data import DataLoader
from utils import visualize_as_video
from as_dataloader_revision import AorticStenosisDataset

droot = r"D:\Datasets\as_tom"
as_data = AorticStenosisDataset(dataset_root=droot, 
                                split='train',
                                view='all',
                                return_info=True,
                                flip_rate=0.0,
                                label_scheme_name='all',
                                normalize=True)

as_loader = DataLoader(as_data, batch_size=1, shuffle=False)

x_fr = [] 
y_fr = []
cycles = []
means = []
vs = []

# cine, cine_orig, label, data_info = next(iter(as_loader))
# vo, va = visualize_as_video(cine.squeeze().numpy(), cine_orig.squeeze().numpy(),
#                             data_info['path'])
multiple=False
create_videos=False
if multiple:
    i=0
    for cine, label_1, label_2, data_info, cine_orig in as_loader:
        xfr = data_info['window_length']
        yfr = data_info['original_length']
        approximate_cycles = xfr*data_info['heart_rate']*data_info['frame_time']/60000
        print('reading image {0}, cycles={1:.3f}'.format(i, approximate_cycles.numpy()[0]))
        x_fr.append(xfr.numpy()[0]) 
        y_fr.append(yfr.numpy()[0])
        cycles.append(approximate_cycles.numpy()[0])
        if create_videos and i % 50 == 0: 
            print('drawing image {0}, path={1}'.format(i, data_info['path']))
            vid_name = 'data_vis_video_{0}.gif'.format(int(i/50))
            vo, va = visualize_as_video(cine.squeeze().numpy(), 
                                        cine_orig.squeeze().numpy(),
                                        data_info,
                                        output_path=vid_name)
        i=i+1
        m = torch.mean(cine)
        means.append(m.item())
        vs.append(torch.mean(torch.square(cine-m)).item())
    import matplotlib.pyplot as plt
    plt.hist(vs)
else:
    cine, label_1, label_2, data_info, cine_orig = next(iter(as_loader))
    #m = torch.mean(cine)
    #vs= torch.mean(torch.square(cine-m)).item()
    # vo, va = visualize_as_video(cine.squeeze().numpy(), 
    #                             cine_orig.squeeze().numpy(),
    #                             data_info)
    