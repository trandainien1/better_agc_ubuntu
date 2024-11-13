
# Generate heatmap of our method
prediction, better_agc_heatmap, output_truth  = BetterCAM_Method.generate(image)

agc_scores = []

for i in range(better_agc_heatmap.size(1)):  
# Loop over the first dimension (12)
for j in range(better_agc_heatmap.size(2)): # Loop over the second dimension (12)
    tensor_heatmap = transforms.Resize((224, 224))(better_agc_heatmap[0][i][j])
    tensor_heatmap = (tensor_heatmap - tensor_heatmap.min())/(tensor_heatmap.max()-tensor_heatmap.min())
    tensor_heatmap = tensor_heatmap.unsqueeze(0).to(device)
    
    tensor_img = transformed_img.unsqueeze(0).to(device)

    model.zero_grad()
    # print(tensor_img.shape)
    # print(tensor_heatmap.shape)
    output_mask = model(tensor_img * tensor_heatmap)
    
    agc_score = torch.sum(torch.abs((output_mask - output_truth[:, prediction[0]])))
    # agc_score = output_mask[0, prediction[0]] - output_truth[0, prediction[0]]
    agc_scores.append(agc_score.detach().cpu().numpy())

masks = better_agc_heatmap[0]
e_x = np.exp(agc_scores - np.max(agc_scores)) 
agc_scores = e_x / e_x.sum(axis=0)
agc_scores = agc_scores.reshape(masks.shape[0], masks.shape[1])

my_cam = (agc_scores[:, :, None, None, None] * masks.detach().cpu().numpy()).sum(axis=(0, 1))

my_cam = transforms.Resize((224, 224))(torch.from_numpy(my_cam))
my_cam = (my_cam - my_cam.min())/(my_cam.max()-my_cam.min())
my_cam = my_cam.detach().cpu().numpy()
my_cam = np.transpose(my_cam, (1, 2, 0))  