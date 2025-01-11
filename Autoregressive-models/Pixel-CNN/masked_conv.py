import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    """
    
    The masking follows a raster scan order (left-to-right, top-to-bottom), with two types:
    - Type A: Used in the first layer, masks the center pixel and all future pixels
    - Type B: Used in subsequent layers, only masks future pixels
    
    """
    def __init__(self, in_c, out_c, kernel_size, mask_type='A', **kwargs):
        super().__init__(in_c, out_c, kernel_size, **kwargs)
        
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        
       
        self.register_buffer('mask', self.create_mask(kernel_size, mask_type))
        
        self.apply_mask()
    
    def create_mask(self, kernel_size, mask_type):
       
        height, width = kernel_size, kernel_size
        mask = torch.ones(self.weight.shape)  
        
        h_center, w_center = height // 2, width // 2
        
        for h in range(height):
            for w in range(width):
                if h > h_center:
                    mask[:, :, h, w] = 0
                elif w > w_center:
                    mask[:, :, h, w] = 0
                elif w == w_center and mask_type == 'A':
                    mask[:, :, h, w] = 0
        
        return mask
    
    def apply_mask(self):
        
        self.weight.data *= self.mask
    
    def forward(self, x):

        self.apply_mask()  
        return super().forward(x)