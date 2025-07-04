import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import cv2
from PIL import Image
import time
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import pickle
import torch.multiprocessing as mp
import argparse
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import gc




import cv2
import os
from tkinter import simpledialog, messagebox, Tk

def capture_face_images(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    捕获人脸图像并按给定比例分配到训练、验证和测试集中
    
    参数:
        data_dir: 数据库根目录
        train_ratio: 分配到训练集的比例
        val_ratio: 分配到验证集的比例
        test_ratio: 分配到测试集的比例
    """
    # 创建Tkinter根窗口（用于对话框）
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 获取人名
    name = simpledialog.askstring("输入", "请输入姓名:")
    if not name:
        print("未输入姓名。退出。")
        return
    
    # 创建train/val/test目录下的人名目录
    train_dir = os.path.join(data_dir, 'train', name)
    val_dir = os.path.join(data_dir, 'val', name)
    test_dir = os.path.join(data_dir, 'test', name)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 显示信息消息
    messagebox.showinfo("提示", "点击'确定'后，摄像头将开始捕获图像。")
    
    # 开始摄像头捕获
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0
    max_images = 100  # 最多捕获100张图像

    images = []  # 存储所有捕获的图像

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("无法捕获图像。退出。")
            break

        # 显示当前捕获进度
        progress_frame = frame.copy()
        cv2.putText(progress_frame, f"已捕获: {count}/{max_images}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("捕获人脸中", progress_frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))  # 统一调整大小
            # 仅存储图像，稍后再保存到不同目录
            images.append(face)
            count += 1
            break  # 每一帧只处理一个人脸

        # 按"q"键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 如果没有捕获到图像，提前返回
    if not images:
        messagebox.showinfo("警告", "未捕获到任何人脸图像!")
        return

    # 计算每个集合应有的图像数量
    num_train = int(len(images) * train_ratio)
    num_val = int(len(images) * val_ratio)
    num_test = len(images) - num_train - num_val
    
    # 打乱图像顺序
    import random
    random.shuffle(images)
    
    # 分配到训练集
    for i in range(num_train):
        cv2.imwrite(f"{train_dir}/{i}.jpg", images[i])
    
    # 分配到验证集
    for i in range(num_val):
        cv2.imwrite(f"{val_dir}/{i}.jpg", images[num_train + i])
    
    # 分配到测试集
    for i in range(num_test):
        cv2.imwrite(f"{test_dir}/{i}.jpg", images[num_train + num_val + i])
    
    # 显示成功消息
    msg = f"人脸捕获成功!\n训练集: {num_train}张\n验证集: {num_val}张\n测试集: {num_test}张"
    messagebox.showinfo("成功", msg)
    print(msg)

class FaceDataset(Dataset):
    def __init__(self, image_folder, mtcnn, transform=None):
        self.mtcnn = mtcnn
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # Search folders for images of names
        for class_idx, class_name in enumerate(os.listdir(image_folder)):
            class_path = os.path.join(image_folder, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                self.class_to_idx[class_name] = class_idx
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, class_idx))
        
        # Print class information for debugging
        print(f"Found {len(self.classes)} classes in {image_folder}: {', '.join(self.classes)}")
        print(f"Loaded {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # CV2 read images in BGR format
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            print(f"[ERROR] Cannot read image: {img_path}")
            return torch.zeros((3, 160, 160)), class_idx

        # Convert BGR ro RGB as MTCNN uses RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # MTCNN face detection
        face = self.mtcnn(img_pil)
        
        if face is None:
            print(f"[WARNING] No face detected in {img_path}")
            # If no face detected, return next sample
            return self.__getitem__((idx + 1) % len(self.samples))

        # MTCNN already returns tensor, no need to convert to tensor
        if self.transform and isinstance(face, torch.Tensor):
            # Check transform type
            if hasattr(self.mtcnn, 'tensor_transform'):
                # Use tensor-specific transform
                face = self.mtcnn.tensor_transform(face)
            elif hasattr(self.transform, 'transforms'):
                # Find transforms that are not ToTensor and apply
                for t in self.transform.transforms:
                    if not isinstance(t, transforms.ToTensor):
                        try:
                            face = t(face)
                        except Exception as e:
                            print(f"Warning: Transform {t.__class__.__name__} failed: {e}")
        elif self.transform and not isinstance(face, torch.Tensor):
            # If not tensor, apply full transform
            try:
                face = self.transform(face)
            except Exception as e:
                print(f"Warning: Transform failed {img_path}: {e}")
                # Return normalized zero tensor as fallback
                return torch.zeros((3, 160, 160)), class_idx

        return face, class_idx



class FaceRecognitionSystem:
    def __init__(self, database_path='face_database', force_cpu=False):
        # Initialize MTCNN for face detection
        self.device = torch.device('cpu') if force_cpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        if not force_cpu and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20, 
            min_face_size=20,
            thresholds=[0.45, 0.55, 0.55],  # 稍微降低人脸检测阈值
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # 修改为使用不分类的模型，更适合提取人脸特征
        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.resnet.eval() 
        
        # 设置基础路径和子目录
        self.database_path = database_path
        self.train_path = os.path.join(database_path, 'train')
        self.val_path = os.path.join(database_path, 'val')
        self.test_path = os.path.join(database_path, 'test')
        
        # 检查目录是否存在
        self._check_directories()
        
        self.embeddings_dict = {}
        self.names = []
        self.vote_history = {}  
        self.vote_window = 10  # 减小投票窗口，使系统反应更快
        self.vote_threshold = 0.7  # 提高投票阈值
        self.tracking_threshold = 100  # 增加跟踪阈值，提高跟踪容忍度

        # 定义不同的转换流程
        # 1. 用于PIL图像的完整转换
        self.pil_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # 减小旋转角度，避免过度变形
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 减小颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 2. 用于tensor的转换
        self.tensor_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 3. 验证集的转换
        self.pil_val_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 4. 验证集tensor的转换
        self.tensor_val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.transform = self.pil_transform
        self.val_transform = self.pil_val_transform
    
    def _check_directories(self):
        """Check and print dataset directory status"""
        print(f"\nDataset Directory Status:")
        if os.path.exists(self.train_path):
            print(f"✓ Training set directory exists: {self.train_path}")
            train_persons = [d for d in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, d))]
            print(f"  Found {len(train_persons)} person folders: {', '.join(train_persons)}")
        else:
            print(f"✗ Training set directory not found: {self.train_path}")
            
        if os.path.exists(self.val_path):
            print(f"✓ Validation set directory exists: {self.val_path}")
            val_persons = [d for d in os.listdir(self.val_path) if os.path.isdir(os.path.join(self.val_path, d))]
            print(f"  Found {len(val_persons)} person folders: {', '.join(val_persons)}")
        else:
            print(f"✗ Validation set directory not found: {self.val_path}")
            
        if os.path.exists(self.test_path):
            print(f"✓ Test set directory exists: {self.test_path}")
            test_persons = [d for d in os.listdir(self.test_path) if os.path.isdir(os.path.join(self.test_path, d))]
            print(f"  Found {len(test_persons)} person folders: {', '.join(test_persons)}")
        else:
            print(f"✗ Test set directory not found: {self.test_path}")
            
        print("")
        
    def fine_tune_model(self, num_epochs=50, batch_size=8):
        """Fine-tune model with training set and validate with validation set"""
        try:
            print("Optimizing model...")
        
            # Ensure batch size is greater than 1
            if batch_size <= 1:
                batch_size = 2
                print("Batch size too small, changed to 2")

            # Check if training set directory exists
            if not os.path.exists(self.train_path):
                print(f"Training set directory not found: {self.train_path}")
                return False
            
            # Combine mtcnn with transform
            self.mtcnn.tensor_transform = self.tensor_transform
                
            # Load training dataset
            train_dataset = FaceDataset(self.train_path, self.mtcnn, self.transform)
            if len(train_dataset) == 0:
                print("Training set is empty. Please check if directory contains images")
                return False
            
            self.names = train_dataset.classes
            num_classes = len(train_dataset.classes)
            print(f"Detected {num_classes} classes, training set has {len(train_dataset)} images")
        
            if num_classes < 2:
                print("Training requires images of at least 2 people")
                return False
        
            # Check validation set directory
            val_dataset = None
            if os.path.exists(self.val_path):
                # Set tensor transform for validation set
                self.mtcnn.tensor_transform = self.tensor_val_transform
                # Use the exact same transform for validation to prevent training/validation gap
                val_dataset = FaceDataset(self.val_path, self.mtcnn, self.val_transform)
                print(f"Validation set has {len(val_dataset)} images")
            else:
                print(f"Validation set directory not found: {self.val_path}, will not use validation set")
            
            # Modify last layer to adapt to new class count
            self.resnet = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(train_dataset.classes))
            self.resnet = self.resnet.to(self.device)
            self.resnet.train()
        
            # Create data loader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam([
                {'params': self.resnet.logits.parameters(), 'lr': 0.001},
                {'params': self.resnet.block8.parameters(), 'lr': 0.0001},
                {'params': self.resnet.mixed_7a.parameters(), 'lr': 0.0001}
            ])
        
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.5, patience=3, 
                                                       verbose=True)
        
            best_accuracy = 0
            patience = 10  # Increase patience value
            no_improve = 0
            model_save_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
        
            # Add history tracking for plotting
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'epochs': []
            }
            
            # Training loop
            for epoch in range(num_epochs):
                try:
                    # Training phase
                    self.resnet.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        try:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                            optimizer.zero_grad()
                            outputs = self.resnet(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                    
                            running_loss += loss.item()
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()
                    
                            if batch_idx % 5 == 0:  # Print progress every 5 batches
                                print(f'Batch [{batch_idx+1}/{len(train_loader)}]  '
                                      f'Loss: {loss.item():.4f}  '
                                      f'Accuracy: {100.*correct/total:.2f}%')
                                
                            # Clear cache to reduce CUDA memory usage
                            if torch.cuda.is_available() and batch_idx % 10 == 0:
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            print(f"Error processing batch {batch_idx}: {e}")
                            continue
                
                    train_loss = running_loss / len(train_loader)
                    train_accuracy = 100. * correct / total
                    print(f'Epoch {epoch+1}/{num_epochs}:  Training Loss: {train_loss:.4f},  Training Accuracy: {train_accuracy:.2f}%')
                    
                    # Add to history
                    history['train_loss'].append(train_loss)
                    history['train_acc'].append(train_accuracy)
                    history['epochs'].append(epoch+1)
                    
                    # Validation phase
                    val_accuracy = 0
                    if val_loader:
                        try:
                            self.resnet.eval()
                            val_loss = 0.0
                            correct = 0
                            total = 0
                    
                            with torch.no_grad():
                                for batch_idx, (inputs, targets) in enumerate(val_loader):
                                    try:
                                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                            
                                        outputs = self.resnet(inputs)
                                        loss = criterion(outputs, targets)
                            
                                        val_loss += loss.item()
                                        _, predicted = outputs.max(1)
                                        total += targets.size(0)
                                        correct += predicted.eq(targets).sum().item()
                                    except Exception as e:
                                        print(f"Error processing batch {batch_idx}: {e}")
                                        continue
                            
                            if total > 0:
                                val_loss = val_loss / len(val_loader)
                                val_accuracy = 100. * correct / total
                                print(f'Validation Loss: {val_loss:.4f},  Validation Accuracy: {val_accuracy:.2f}%')
                                
                                # Add to history
                                history['val_loss'].append(val_loss)
                                history['val_acc'].append(val_accuracy)
                            else:
                                print("No valid samples in validation set")
                                val_accuracy = 0
                                
                                # Add empty values for consistency
                                history['val_loss'].append(None)
                                history['val_acc'].append(None)
                                
                            # Update learning rate
                            scheduler.step(val_accuracy)
                        except Exception as e:
                            print(f"Error during validation: {e}")
                            val_accuracy = 0
                            
                            # Add empty values for consistency
                            history['val_loss'].append(None)
                            history['val_acc'].append(None)
                    else:
                        # If no validation set, use training accuracy to update learning rate
                        scheduler.step(train_accuracy)
                        val_accuracy = train_accuracy
                
                        # Add empty values for consistency
                        history['val_loss'].append(None)
                        history['val_acc'].append(None)
                    
                    # Save best model
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.resnet.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'accuracy': val_accuracy,
                            'class_names': self.names,
                            'history': history  # Save training history
                        }, model_save_path)
                        no_improve = 0
                        print(f"Saved new best model, accuracy: {val_accuracy:.2f}%")
                    else:
                        no_improve += 1
                    
                    # Early stopping mechanism
                    if no_improve >= patience:
                        print(f"Continuous {patience} epochs without improvement, stopping training")
                        break
                    
                    # Plot intermediate learning curves every 5 epochs
                    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                        interim_history_path = os.path.join(os.path.dirname(__file__), 'interim_history.pkl')
                        with open(interim_history_path, 'wb') as f:
                            pickle.dump(history, f)
                        self._plot_training_history(history)
                        
                    # Clear CUDA cache after each epoch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    print(f"Epoch {epoch+1} Training error: {e}")
                    import traceback
                    traceback.print_exc()
                    
            print("Model fine-tuning completed!")
            
            # Save final training history for evaluation
            final_history_path = os.path.join(os.path.dirname(__file__), 'training_history.pkl')
            with open(final_history_path, 'wb') as f:
                pickle.dump(history, f)
                
            # Plot final learning curves
            self._plot_training_history(history)
            
            # Load best model
            try:
                if os.path.exists(model_save_path):
                    checkpoint = torch.load(model_save_path)
                    self.resnet.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded best model, accuracy: {checkpoint['accuracy']:.2f}%")
                    self.names = checkpoint['class_names']
                
                    # Evaluate model on test set
                    if os.path.exists(self.test_path):
                        print("Starting to evaluate model on test set...")
                        self.evaluate_model(model_save_path)
                else:
                    print("Warning: Best model file not found")
            except Exception as e:
                print(f"Error loading or evaluating best model: {e}")
                
            # Switch to evaluation mode
            self.resnet.eval()
            
            # Clear CUDA cache at the end of training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def evaluate_model(self, model_path=None):
        """Evaluate model performance on test set"""
        try:
            if not os.path.exists(self.test_path):
                print(f"Test set directory not found: {self.test_path}")
                return False
        
            # Load model (if model path provided)
            if model_path and os.path.exists(model_path):
                # First load checkpoint to get class information
                checkpoint = torch.load(model_path)
                class_names = checkpoint['class_names']
                
                # Reinitialize model to match saved class count
                print(f"Loading model, class count: {len(class_names)}")
                self.resnet = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(class_names))
                self.resnet = self.resnet.to(self.device)
                
                # Now load weights
                self.resnet.load_state_dict(checkpoint['model_state_dict'])
                self.names = class_names
                print(f"Model loaded: {model_path}")
                
                # Load training history
                history = checkpoint.get('history', None)
                if history:
                    self._plot_training_history(history)
                else:
                    # Try to load from separate file
                    history_path = os.path.join(os.path.dirname(__file__), 'training_history.pkl')
                    if os.path.exists(history_path):
                        with open(history_path, 'rb') as f:
                            history = pickle.load(f)
                        self._plot_training_history(history)
                    else:
                        print("Warning: No training history found for plotting learning curves")
            
            # Set tensor transform for test set
            self.mtcnn.tensor_transform = self.tensor_val_transform
            
            # Load test dataset
            test_dataset = FaceDataset(self.test_path, self.mtcnn, self.val_transform)
            if len(test_dataset) == 0:
                print("Test set is empty. Please check if directory contains images")
                return False
        
            # Check if test set classes match training set classes
            model_class_set = set(self.names)
            test_class_set = set(test_dataset.classes)
            
            print(f"Found {len(test_dataset.classes)} classes in {self.test_path}: {', '.join(test_dataset.classes)}")
            print(f"Loaded {len(test_dataset)} images")
            
            # Create mapping from test set class to model class
            test_to_model_idx = {}
            for test_label in test_dataset.class_to_idx:
                if test_label in self.names:
                    test_to_model_idx[test_dataset.class_to_idx[test_label]] = self.names.index(test_label)
                else:
                    print(f"Warning: Class '{test_label}' in test set does not exist in model, will be ignored")
            
            if test_class_set != model_class_set:
                print("Warning: Classes in test set do not match classes used during model training")
                print(f"Model classes: {sorted(model_class_set)}")
                print(f"Test set classes: {sorted(test_class_set)}")
            
            print(f"Evaluating model on test set ({len(test_dataset)} images)...")
        
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
        
            # Switch to evaluation mode
            self.resnet.eval()
        
            correct = 0
            total = 0
            test_loss = 0
            criterion = nn.CrossEntropyLoss()
            
            # For confusion matrix
            all_predictions = []
            all_targets = []
            
            # Create statistics for each class
            class_correct = {cls_name: 0 for cls_name in self.names}
            class_total = {cls_name: 0 for cls_name in self.names}
        
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    try:
                        # Check if labels are valid
                        valid_indices = []
                        valid_labels = []
                        
                        # Map test set labels to model class labels
                        for j, label in enumerate(labels):
                            label_item = label.item()
                            if label_item in test_to_model_idx:
                                valid_indices.append(j)
                                valid_labels.append(test_to_model_idx[label_item])
                        
                        if not valid_indices:  # Skip batch if no valid samples
                            print(f"No valid samples in batch {i}, skipping")
                            continue
                        
                        # Only use valid images and labels
                        valid_images = images[valid_indices].to(self.device)
                        valid_labels = torch.tensor(valid_labels, device=self.device)
                        
                        outputs = self.resnet(valid_images)
                        loss = criterion(outputs, valid_labels)
                        test_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        
                        # Store for confusion matrix
                        all_predictions.extend(predicted.cpu().numpy())
                        all_targets.extend(valid_labels.cpu().numpy())
                        
                        for j, label in enumerate(valid_labels):
                            label_idx = label.item()
                            cls_name = self.names[label_idx]
                            class_total[cls_name] += 1
                            if predicted[j] == label:
                                class_correct[cls_name] += 1
                                correct += 1
                        total += len(valid_labels)
                    except Exception as e:
                        print(f"Error processing batch {i}: {str(e)}")
                        continue
            
            # Avoid division by zero
            if total == 0:
                print("No valid test samples, cannot calculate accuracy")
                return False
            
            # Calculate overall accuracy
            accuracy = 100 * correct / total
            print(f"\nTest set accuracy: {accuracy:.2f}%")
            
            # Output accuracy for each class
            print("\nAccuracy by class:")
            for cls_name in self.names:
                if class_total[cls_name] > 0:
                    accuracy = 100 * class_correct[cls_name] / class_total[cls_name]
                    print(f"{cls_name}: {accuracy:.2f}% ({class_correct[cls_name]}/{class_total[cls_name]})")
                else:
                    print(f"{cls_name}: No test samples")
            
            # Generate and plot confusion matrix
            self._plot_confusion_matrix(all_targets, all_predictions)
            
            # Generate classification report
            report = classification_report(all_targets, all_predictions, 
                                          target_names=self.names, 
                                          digits=3, 
                                          output_dict=True)
            
            print("\nDetailed Classification Report:")
            for cls_name in self.names:
                if cls_name in report:
                    print(f"{cls_name}:")
                    print(f"  Precision: {report[cls_name]['precision']:.3f}")
                    print(f"  Recall: {report[cls_name]['recall']:.3f}")
                    print(f"  F1-score: {report[cls_name]['f1-score']:.3f}")
            
            print(f"\nMacro avg F1-score: {report['macro avg']['f1-score']:.3f}")
            print(f"Weighted avg F1-score: {report['weighted avg']['f1-score']:.3f}")
            
            return accuracy
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0
        
    def _plot_training_history(self, history):
        """Plot training and validation loss and accuracy curves"""
        try:
            # Filter out invalid values, especially None values
            if not history['epochs'] or len(history['epochs']) <= 1:
                print("Not enough training history data points for plotting learning curves")
                return
                
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot training & validation loss
            ax1.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
            if any(v is not None for v in history['val_loss']):
                # Filter out None values
                val_epochs = [e for e, v in zip(history['epochs'], history['val_loss']) if v is not None]
                val_loss = [v for v in history['val_loss'] if v is not None]
                if val_epochs:
                    ax1.plot(val_epochs, val_loss, 'r-', label='Validation Loss')
            
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot training & validation accuracy
            ax2.plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy')
            if any(v is not None for v in history['val_acc']):
                # Filter out None values
                val_epochs = [e for e, v in zip(history['epochs'], history['val_acc']) if v is not None]
                val_acc = [v for v in history['val_acc'] if v is not None]
                if val_epochs:
                    ax2.plot(val_epochs, val_acc, 'r-', label='Validation Accuracy')
            
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(os.path.dirname(__file__), 'learning_curves.png')
            plt.savefig(fig_path)
            print(f"Learning curves saved to {fig_path}")
            
            # Show potential overfitting or underfitting
            if any(v is not None for v in history['val_loss']):
                # Calculate gap between training and validation at the end of training
                final_train_loss = history['train_loss'][-1]
                final_val_loss = next((v for v in reversed(history['val_loss']) if v is not None), None)
                
                if final_val_loss is not None:
                    gap = final_val_loss - final_train_loss
                    
                    print("\nModel Fitting Analysis:")
                    if gap > 0.3:  # Threshold for significant gap
                        print("Warning: Potential overfitting detected.")
                        print(f"Training loss: {final_train_loss:.4f}, Validation loss: {final_val_loss:.4f}")
                        print("Gap: {:.4f}".format(gap))
                        print("Recommendations:")
                        print("- Increase data augmentation")
                        print("- Add dropout or regularization")
                        print("- Reduce model complexity")
                        print("- Use early stopping (already implemented)")
                    elif gap < -0.1:  # Negative gap might indicate validation set is easier
                        print("Unusual pattern: Validation loss lower than training loss.")
                        print(f"Training loss: {final_train_loss:.4f}, Validation loss: {final_val_loss:.4f}")
                        print("Gap: {:.4f}".format(gap))
                        print("This might indicate:")
                        print("- Validation set is easier than training set")
                        print("- Different regularization between training and validation")
                        print("- Training augmentations make training harder")
                    else:
                        print("No significant overfitting detected.")
                        
                    # Check for underfitting
                    if final_train_loss > 0.5:  # Relatively high loss
                        print("\nWarning: Potential underfitting detected.")
                        print(f"Training loss: {final_train_loss:.4f} is still high")
                        print("Recommendations:")
                        print("- Train for more epochs")
                        print("- Increase model complexity")
                        print("- Reduce regularization")
                        print("- Use a higher learning rate")
            
            plt.close(fig)  # Close the figure to free memory
            
        except Exception as e:
            print(f"Error plotting training history: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot with seaborn for better styling
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.names,
                       yticklabels=self.names)
            
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save figure
            cm_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
            
            # Calculate per-class metrics from confusion matrix
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            
            # Per-class sensitivity (recall)
            sensitivity = TP/(TP+FN)
            # Per-class specificity
            specificity = TN/(TN+FP)
            
            # Print metrics
            print("\nDetailed Metrics from Confusion Matrix:")
            for i, name in enumerate(self.names):
                if i < len(sensitivity) and i < len(specificity):
                    print(f"{name}:")
                    print(f"  Sensitivity (Recall): {sensitivity[i]:.3f}")
                    print(f"  Specificity: {specificity[i]:.3f}")
            
            plt.close()  # Close the figure to free memory
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {str(e)}")
            import traceback
            traceback.print_exc()

    def evaluate_unlabeled_test(self, model_path=None, unlabeled_test_path=None):
        """Evaluate model on unlabeled test set and generate predictions"""
        try:
            # Use specified path or default path
            test_path = unlabeled_test_path if unlabeled_test_path else os.path.join(os.path.dirname(self.train_path), "unlabeled")
            
            if not os.path.exists(test_path):
                print(f"Unlabeled test set directory not found: {test_path}")
                print("\nPlease follow these steps to add unlabeled test images:")
                print(f"1. Create directory: mkdir -p {test_path}")
                print(f"2. Copy test images to directory: cp /path/to/your/images/* {test_path}/")
                print("3. Run this command again: python face_detection.py --unlabeled-test")
                print("\nNote: Images must contain clear faces, supported formats: .jpg, .jpeg, .png")
                return False
        
            # Load model (if model path provided)
            if model_path and os.path.exists(model_path):
                # First load checkpoint to get class information
                checkpoint = torch.load(model_path)
                class_names = checkpoint['class_names']
                
                # Reinitialize model to match saved class count
                print(f"Loading model, class count: {len(class_names)}")
                self.resnet = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(class_names))
                self.resnet = self.resnet.to(self.device)
                
                # Now load weights
                self.resnet.load_state_dict(checkpoint['model_state_dict'])
                self.names = class_names
                print(f"Model loaded: {model_path}")
        
            # Set tensor transform for test set
            self.mtcnn.tensor_transform = self.tensor_val_transform
            
            # Switch to evaluation mode
            self.resnet.eval()
        
            # Check if test set is marked (organized by folder)
            test_is_marked = False
            test_dirs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
            
            if test_dirs:
                print(f"Test set appears marked (contains subfolders): {test_dirs}")
                print("Please use evaluate_model method to evaluate labeled test set")
                print("If you really want to evaluate unlabeled images, please put them directly in the test directory, not in subfolders")
                return False
            
            # Get all images from test set
            test_images = []
            for img_name in os.listdir(test_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_images.append(os.path.join(test_path, img_name))
        
            if not test_images:
                print("Test set is empty. Please check if directory contains images")
                return False
        
            print(f"Evaluating model on unlabeled test set ({len(test_images)} images)...")
        
            results = []
        
            for img_path in test_images:
                try:
                    # Read image
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        print(f"[ERROR] Image read failed: {img_path}")
                        continue
                
                    # Convert to RGB
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                
                    # Face detection and recognition
                    face = self.mtcnn(img_pil)
                
                    if face is not None:
                        with torch.no_grad():
                            face = face.unsqueeze(0).to(self.device)
                            # Ensure in evaluation mode
                            outputs = self.resnet(face)
                            # Get prediction results
                            probs = torch.softmax(outputs, dim=1)[0]
                            
                            # Get top 3 most likely prediction results
                            values, indices = probs.topk(min(3, len(self.names)))
                            
                            # Main prediction result
                            predicted_idx = indices[0].item()
                            if predicted_idx < len(self.names):
                                predicted_name = self.names[predicted_idx]
                            else:
                                print(f"[WARNING] Predicted index out of range: {predicted_idx} >= {len(self.names)}")
                                predicted_name = "Unknown"
                            confidence = values[0].item()
                            
                            # Secondary prediction results
                            top3_predictions = []
                            for i in range(min(3, len(self.names))):
                                idx = indices[i].item()
                                if idx < len(self.names):
                                    name = self.names[idx]
                                else:
                                    name = "Unknown"
                                top3_predictions.append({
                                    "name": name,
                                    "confidence": values[i].item()
                                })
                    
                        results.append({
                            "image": os.path.basename(img_path),
                            "predicted": predicted_name,
                            "confidence": confidence,
                            "top3": top3_predictions
                        })
                    
                        # Format output top 3 predictions and confidence
                        top3_str = ", ".join([f"{p['name']}({p['confidence']:.2f})" for p in top3_predictions])
                        print(f"[{os.path.basename(img_path)}] Predicted: {predicted_name}, Confidence: {confidence:.4f}")
                        print(f"   Top 3 Predictions: {top3_str}")
                    else:
                        print(f"[WARNING] No valid face detected in {img_path}")
                except Exception as e:
                    print(f"[ERROR] Error processing {img_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            if not results:
                print("No images successfully processed")
                return []
                
            # Print summary
            try:
                print("\nPrediction results summary:")
                class_counts = {}
                for result in results:
                    predicted = result["predicted"]
                    if predicted not in class_counts:
                        class_counts[predicted] = 0
                    class_counts[predicted] += 1
        
                for name, count in class_counts.items():
                    print(f"{name}: {count} images ({count/len(results)*100:.2f}%)")
            except Exception as e:
                print(f"Error during summary: {e}")
        
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def create_database(self):
        """从训练集创建人脸嵌入数据库"""
        train_path = self.train_path
        
        if not os.path.exists(train_path):
            print(f"Training directory not found: {train_path}")
            return False
            
        dataset = datasets.ImageFolder(train_path)
        self.names = dataset.classes
        
        print(f"Creating database from training set: {train_path}")
        print(f"Detected people: {', '.join(self.names)}")
        
        # Create feature extraction model, ensure output is 512D feature vector
        feature_extractor = InceptionResnetV1(pretrained='vggface2', classify=False)
        feature_extractor.eval()
        feature_extractor.to(self.device)
        
        # Create a list for each person to store all embeddings
        temp_embeddings = {name: [] for name in self.names}
        
        for img_path, class_idx in dataset.imgs:
            try:
                # Use OpenCV to read image (BGR format)
                img_bgr = cv2.imread(img_path)
                
                if img_bgr is None:
                    print(f"[ERROR] Image read failed: {img_path}")
                    continue
                
                # Convert BGR to RGB (MTCNN requires RGB format)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # Face detection
                face = self.mtcnn(img_pil)
                
                if face is not None:
                    with torch.no_grad():
                        face = face.unsqueeze(0).to(self.device)
                        # Use feature extraction model to get 512D feature vector
                        embedding = feature_extractor(face).cpu().numpy()[0]
                    
                    person_name = self.names[class_idx]
                    temp_embeddings[person_name].append(embedding)
                    print(f"[SUCCESS] Embedding generated for {person_name}: {img_path}")
                else:
                    print(f"[WARNING] No valid face detected in {img_path}")
            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {str(e)}")
                continue

        # Calculate average embedding and threshold for each person
        for person in self.names:
            if temp_embeddings[person]:
                embeddings = np.array(temp_embeddings[person])
                mean_embedding = np.mean(embeddings, axis=0)
                
                distances = []
                for emb in embeddings:
                    dist = np.linalg.norm(emb - mean_embedding)
                    distances.append(dist)
                
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                
                # Use balanced threshold
                threshold = max(mean_dist + 2.2 * std_dist, 2.2)  # Changed from 2.0 to 2.2
                
                self.embeddings_dict[person] = mean_embedding
                self.embeddings_dict[f"{person}_threshold"] = threshold
                print(f"[INFO] Threshold set for {person}: {threshold:.3f} (Average: {mean_dist:.3f}, Standard Deviation: {std_dist:.3f})")
            else:
                print(f"[ERROR] No embeddings found for '{person}'. Adding default embedding.")
                self.embeddings_dict[person] = np.zeros(512)
                self.embeddings_dict[f"{person}_threshold"] = 2.2  # Changed from 2.0 to 2.2

        with open('embeddings.pkl', 'wb') as f:
            pickle.dump((self.embeddings_dict, self.names), f)
            
        return True

    def load_database(self):
        """Load pre-computed embeddings from file"""
        if os.path.exists('embeddings.pkl'):
            with open('embeddings.pkl', 'rb') as f:
                self.embeddings_dict, self.names = pickle.load(f)
            return True
        return False
        
    def recognize_face(self, face_embedding):
        distances = {}
        min_dist = float('inf')
        best_match = None
        
        for person in self.names:
            embedding = self.embeddings_dict[person]
            dist = np.linalg.norm(face_embedding - embedding)
            person_threshold = self.embeddings_dict.get(f"{person}_threshold", 2.2)  # Slightly increase default threshold
            distances[person] = (dist, person_threshold)
            
            if dist < min_dist:
                min_dist = dist
                best_match = (person, person_threshold)
        
        if best_match:
            person_name, threshold = best_match
            # Use balanced confidence calculation
            confidence = max(0, 1 - min_dist/(threshold * 1.0))  # Use original threshold, no increase or decrease
            # Use balanced decision threshold
            identity = person_name if min_dist <= threshold * 1.0 else "Unknown"  # Changed from 0.9 to 1.0
        else:
            identity = "Unknown"
            confidence = 0
            min_dist = float('inf')
        
        return identity, confidence, min_dist
        
    def update_vote_history(self, face_id, name, distance):
        """Update vote history"""
        if face_id not in self.vote_history:
            self.vote_history[face_id] = []
        
        self.vote_history[face_id].append((name, distance))
        if len(self.vote_history[face_id]) > self.vote_window:
            self.vote_history[face_id].pop(0)
            
    def get_voted_identity(self, face_id):
        if face_id not in self.vote_history:
            return "Unknown", 0
        
        vote_count = {}
        total_weights = {}
        decay_factor = 0.87  # Slightly increase decay factor, balance history frame impact
        
        # Calculate time-weighted vote
        for i, (name, dist) in enumerate(self.vote_history[face_id]):
            weight = decay_factor ** (len(self.vote_history[face_id]) - i - 1)
            if name not in vote_count:
                vote_count[name] = {"count": 0, "total_dist": 0, "weighted_count": 0}
            vote_count[name]["count"] += 1
            vote_count[name]["total_dist"] += dist * weight
            vote_count[name]["weighted_count"] += weight
        
        max_votes = 0
        best_identity = "Unknown"
        best_confidence = 0
        
        for name, info in vote_count.items():
            if info["weighted_count"] > max_votes:
                max_votes = info["weighted_count"]
                best_identity = name
                avg_dist = info["total_dist"] / info["weighted_count"]
                threshold = self.embeddings_dict.get(f"{name}_threshold", 2.2)
                # Use balanced confidence calculation
                best_confidence = max(0, 1 - avg_dist/threshold)
        
        # Use balanced vote threshold
        vote_threshold = self.vote_window * 0.4  # Changed from 0.5 to 0.4
        if max_votes < vote_threshold:
            return "Unknown", 0
            
        return best_identity, best_confidence
        
    def run_live_recognition(self):
        """Run live face recognition using webcam"""
        if not self.load_database():
            print("No database found. Please create database first.")
            return
            
        print("\nAll people in database:")
        for name in self.names:
            print(f"- {name}")
        print("\nStarting real-time recognition, press 'q' to exit...")
        
        # Create feature extraction model, ensure output is 512D feature vector instead of classification result
        feature_extractor = InceptionResnetV1(pretrained='vggface2', classify=False)
        feature_extractor.eval()
        feature_extractor.to(self.device)
        
        # Keep original classification model for classification results
        self.resnet.eval()
            
        cap = cv2.VideoCapture(0)
        face_trackers = {}  # Used for tracking different faces
        next_face_id = 0
        
        # Set camera parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 2 != 0:  # Process every other frame for performance
                continue
                
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes, _ = self.mtcnn.detect(Image.fromarray(frame_rgb))
            
            if boxes is not None:
                # Update or create face tracker
                current_faces = []
                
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Ignore too small faces
                    face_width = x2 - x1
                    face_height = y2 - y1
                    if face_width < 60 or face_height < 60:  # Minimum face size
                        continue
                        
                    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Find nearest tracker
                    matched_id = None
                    for face_id, tracker in face_trackers.items():
                        tx1, ty1, tx2, ty2 = tracker["box"]
                        tracker_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
                        distance = np.sqrt((face_center[0] - tracker_center[0])**2 + 
                                        (face_center[1] - tracker_center[1])**2)
                        if distance < self.tracking_threshold:  # Use threshold from class variable
                            matched_id = face_id
                            break
                    
                    if matched_id is None:
                        matched_id = next_face_id
                        face_trackers[matched_id] = {"box": None}
                        next_face_id += 1
                        
                    face_trackers[matched_id]["box"] = (x1, y1, x2, y2)
                    current_faces.append(matched_id)
                    
                    # Extract face and get embedding
                    face = frame_rgb[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                        
                    face_pil = Image.fromarray(face)
                    face_tensor = self.mtcnn(face_pil)
                    
                    if face_tensor is not None:
                        # Use feature extraction model to get 512D feature vector
                        with torch.no_grad():
                            face_tensor = face_tensor.unsqueeze(0)
                            if torch.cuda.is_available():
                                face_tensor = face_tensor.cuda()
                            # Use feature extraction model, not classification model
                            embedding = feature_extractor(face_tensor).cpu().numpy()[0]
                            
                        # Recognize face
                        name, confidence, distance = self.recognize_face(embedding)
                        
                        # Update vote history
                        self.update_vote_history(matched_id, name, distance)
                        
                        # Get vote results
                        voted_name, voted_confidence = self.get_voted_identity(matched_id)
                        
                        # Draw bounding box and name
                        color = (0, 255, 0) if voted_name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        text = f"{voted_name} ({voted_confidence:.2f})"
                        cv2.putText(frame, text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Clean up face trackers that no longer appear
                face_trackers = {k: v for k, v in face_trackers.items() if k in current_faces}
            
            # Add all names to video frame
            y_offset = 30
            for name in self.names:
                cv2.putText(frame, name, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 20
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def organize_test_set(face_system, test_dir, output_dir, model_path=None):
    """Automatically classify unlabeled test set images into corresponding person folders"""
    if not os.path.exists(test_dir):
        print(f"Test set directory not found: {test_dir}")
        return False
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model (if model path provided)
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        face_system.resnet.load_state_dict(checkpoint['model_state_dict'])
        face_system.names = checkpoint['class_names']
    
    # Switch to evaluation mode
    face_system.resnet.eval()
    
    # Create folder for each person
    for name in face_system.names:
        person_dir = os.path.join(output_dir, name)
        os.makedirs(person_dir, exist_ok=True)
    
    # Get all images from test set
    test_images = []
    for img_name in os.listdir(test_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(os.path.join(test_dir, img_name))
    
    if not test_images:
        print("Test set is empty. Please check if directory contains images")
        return False
    
    print(f"Classifying {len(test_images)} test set images...")
    
    success_count = 0
    unknown_count = 0
    error_count = 0
    
    for img_path in test_images:
        try:
            # Read image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[ERROR] Image read failed: {img_path}")
                error_count += 1
                continue
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Face detection and recognition
            face = face_system.mtcnn(img_pil)
            
            if face is not None:
                with torch.no_grad():
                    face = face.unsqueeze(0).to(face_system.device)
                    # Ensure in evaluation mode
                    outputs = face_system.resnet(face)
                    # Get prediction results
                    _, predicted = outputs.max(1)
                    predicted_name = face_system.names[predicted.item()]
                    confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
                
                # If confidence too low, mark as unknown
                if confidence < 0.5:
                    print(f"[{os.path.basename(img_path)}] Confidence too low: {confidence:.4f}, Marked as 'Unknown'")
                    unknown_count += 1
                    continue
                
                # Copy image to corresponding person folder
                dest_path = os.path.join(output_dir, predicted_name, os.path.basename(img_path))
                cv2.imwrite(dest_path, img_bgr)
                
                print(f"[{os.path.basename(img_path)}] Predicted: {predicted_name}, Confidence: {confidence:.4f}")
                success_count += 1
            else:
                print(f"[WARNING] No valid face detected in {img_path}")
                unknown_count += 1
        except Exception as e:
            print(f"[ERROR] Error processing {img_path}: {str(e)}")
            error_count += 1
    
    # Print summary
    print("\nClassification summary:")
    print(f"Successfully classified: {success_count} images")
    print(f"Unrecognized: {unknown_count} images")
    print(f"Processing errors: {error_count} images")
    
    # Check image count in each person folder
    print("\nImage distribution by person:")
    for name in face_system.names:
        person_dir = os.path.join(output_dir, name)
        if os.path.exists(person_dir):
            image_count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{name}: {image_count} images")
    
    return True

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Set start method for multiprocessing

    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--live", action="store_true", help="Run live face recognition")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage instead of GPU")
    parser.add_argument("--catch", action="store_true", help="Catch images for a new person")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate model on test set")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--unlabeled-test", action="store_true", help="Evaluate model on unlabeled test set")
    parser.add_argument("--database-path", type=str, default='face_database', help="Path to face database")

    args = parser.parse_args()

    face_system = FaceRecognitionSystem(database_path=args.database_path, force_cpu=args.force_cpu)

    try:
        if args.catch:
            data_dir = args.database_path  # Use database root directory, no longer directly save to training set
            os.makedirs(data_dir, exist_ok=True)  # Ensure directory exists
            capture_face_images(data_dir)
            exit(0)

        if args.evaluate:
            # Only evaluate existing model
            model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
            if os.path.exists(model_path):
                print(f"Starting to evaluate model performance on test set...")
                face_system.evaluate_model(model_path)
            else:
                print("No model file found for evaluation")
            exit(0)
        elif args.unlabeled_test:
            # Evaluate model on unlabeled test set
            model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
            if os.path.exists(model_path):
                print(f"Starting to evaluate model performance on unlabeled test set...")
                face_system.evaluate_unlabeled_test(model_path)
            else:
                print("No model file found for evaluation")
            exit(0)
        elif args.live:
            # Directly enter real-time recognition mode
            model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
            if os.path.exists(model_path) and False:  # Temporary disable loading classification model path
                print(f"Loading model and starting real-time recognition...")
                # Load model instead of database
                checkpoint = torch.load(model_path)
                face_system.resnet.load_state_dict(checkpoint['model_state_dict'])
                face_system.names = checkpoint['class_names']
                print(f"Loaded model: {model_path}")
                face_system.run_live_recognition()
            elif face_system.load_database():
                print("Using embedding database for real-time recognition...")
                face_system.run_live_recognition()
            else:
                print("No database found, Creating new database from training data...")
                if face_system.create_database():
                    print("Database created successfully, Starting real-time recognition...")
                    face_system.run_live_recognition()
                else:
                    print("Database creation failed. Please ensure training data set exists.")
                    exit(1)
        else:
            # Normal process: Create database, fine-tune model, then start real-time recognition
            print("\nStarting to execute full process: Create database -> Fine-tune model -> Real-time recognition")
            
            if not os.path.exists(face_system.train_path):
                print(f"Error: Training directory not found: {face_system.train_path}")
                print("Please use --catch parameter to capture face images, or manually create training directory")
                exit(1)
                
            if not face_system.create_database():
                print("Database creation failed. Exiting.")
                exit(1)
            
            print("\nStarting to fine-tune model...")
            if face_system.fine_tune_model(num_epochs=args.epochs, batch_size=args.batch_size):
                print("\nModel fine-tuning completed, Starting real-time recognition...")
                face_system.run_live_recognition()
            else:
                print("Model fine-tuning failed. Exiting.")
                exit(1)
    except Exception as e:
        print(f"Error during main execution: {e}")
        exit(1)
    finally:
        # Clean up CUDA resources at the end of the program
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        