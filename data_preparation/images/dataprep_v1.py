class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                class_images = os.listdir(class_dir)
                self.image_paths.extend([os.path.join(class_dir, img) for img in class_images])
                self.labels.extend([i] * len(class_images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        one_hot_label = one_hot(torch.tensor(label), len(self.classes)).float()


        return image, one_hot_label

def index_splitter(n, splits, seed=13):
    idx = torch.arange(n)
    # Makes the split argument a tensor
    splits_tensor = torch.as_tensor(splits)
    # Finds the correct multiplier, so we don't have
    # to worry about summing up to N (or one)
    multiplier = n / splits_tensor.sum()    
    splits_tensor = (multiplier * splits_tensor).long()
    # If there is a difference, throws at the first split
    # so random_split does not complain
    diff = n - splits_tensor.sum()
    splits_tensor[0] += diff
    # Uses PyTorch random_split to split the indices
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

images_dataset  = CustomDataset(root_dir=PATH, transform=transform)

# image, label = images_dataset[0]  

# reverse_normalize = transforms.Compose([
#     transforms.Normalize(mean=(-0.5 / 0.5,), std=(1.0 / 0.5,))
# ])
# reversed_image_tensor = reverse_normalize(image)

# image_pil = ToPILImage()(reversed_image_tensor)

# plt.imshow(image_pil)
# plt.axis('off')
# plt.show()
# class_name = images_dataset.classes[label]

# print(class_name)

train_idx, val_idx = index_splitter(len(images_dataset), [85, 15])

train_dataset = Subset(images_dataset, train_idx)
val_dataset = Subset(images_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Generated train_loader and val_loader")
