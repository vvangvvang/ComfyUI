import numpy as np
import paddle as pd
import folder_paths
import os
import node_helpers
import torch
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
import json
import hashlib


class Example2:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
        告诉主程序节点的输入参数。
    IS_CHANGED:
        optional method to control when the node is re executed.
        控制节点何时重新执行的可选方法。
    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
        输出元组中每个元素的类型。
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tuple.
        可选项：输出元组中每个输出的名称。
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
        入口点方法的名称。例如，如果“ function=” execute "，则它将运行示例。执行()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        如果该节点是从图中输出结果/图像的输出节点。SaveImage节点就是一个例子。
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        后端在这些输出节点上迭代，如果它们的父图正确连接，则尝试执行它们的所有父节点。
        Assumed to be False if not present.
        如果不存在，则假定为假。
    CATEGORY (`str`):
        The category the node should appear in the UI.
        节点应显示在UI中的类别。
    DEPRECATED (`bool`):
        Indicates whether the node is deprecated. Deprecated nodes are hidden by default in the UI, but remain
        functional in existing workflows that use them.
        指示节点是否已弃用。默认情况下，不建议使用的节点在UI中隐藏，但仍保留
        在使用它们的现有工作流中起作用。
    EXPERIMENTAL (`bool`):
        Indicates whether the node is experimental. Experimental nodes are marked as such in the UI and may be subject to
        significant changes or removal in future versions. Use with caution in production workflows.
        指示节点是否是实验性的。实验节点在UI中标记为实验节点，并且可能受
        未来版本中的重大更改或删除。在生产工作流程中谨慎使用。
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
        入口点方法。此方法的名称必须与属性“ Function ”的值相同。
        例如，如果function=“ execute ”，则此方法的名称必须为“ execute ”，如果“ function=” foo “，则必须为” foo "。
    """
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.
            
            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Second value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "int_field": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number", # Cosmetic only: display as "number" or "slider"
                    "lazy": True # Will only be evaluated if check_lazy_status requires it
                }),
                "float_field": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number",
                    "lazy": True
                }),
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!",
                    "lazy": True
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "paddle"

    def check_lazy_status(self, image, string_field, int_field, float_field, print_to_screen):
        """
            Return a list of input names that need to be evaluated.
            返回需要评估的输入名称列表。
            This function will be called if there are any lazy inputs which have not yet been
            evaluated. As long as you return at least one field which has not yet been evaluated
            (and more exist), this function will be called again once the value of the requested
            field is available.
            如果有任何尚未计算的懒得输入的参数，则将调用此函数。只要您返回至少一个尚未计算的字段（并且存在更多字段），一旦所请求字段的值可用，将再次调用此函数。
            Any evaluated inputs will be passed as arguments to this function. Any unevaluated
            inputs will have the value None.
            任何计算的输入都将作为参数传递给此函数。任何未求值的输入都将具有值“空”。
        """
        if print_to_screen == "enable":
            return ["int_field", "float_field", "string_field"]
        else:
            return []

    def test(self, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        image = float_field - image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
        如果任何输入发生变化，节点将始终重新执行，但
        此方法可用于强制节点再次执行，即使输入没有更改。
        您可以使此节点返回数字或字符串。此值将与上次返回节点时返回的值进行比较。
        如果不同，则将再次执行该节点。
        此方法在LoadImage节点的核心repo中使用，如果图像哈希
        在两次执行之间进行更改时，将再次执行LoadImage节点。
    """
    @classmethod
    def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
        return ""


class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                    "batch_size": ("INT", {"default": 0,"min":1,"max":512,"step":1}), 
                    },
                }

    CATEGORY = "paddle"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image, batch_size):
        image_path = folder_paths.get_annotated_filepath(image)
        print("get image?")
        print(image_path)
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        print("output_image shape",output_image.shape)
        print("output_mask shape",output_mask.shape)

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True



class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "paddle"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }


class RandomDataset:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "randomdataset"

    OUTPUT_NODE = True

    CATEGORY = "paddle/DataSet"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def randomdataset(self, images, filename_prefix="ComfyUI"):
        return { "ui": { "images": results } }

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/hello")
async def get_hello(request):
    print("hello world!")
    return web.json_response("hello")


# A dictionary that contains all nodes you want to export with their names
#包含要导出的所有节点及其名称的字典
# NOTE: names should be globally unique
#注意：名称应全局唯一
NODE_CLASS_MAPPINGS = {
    "Example3": Example2,
    "Load Image": LoadImage,
    "Save Image": SaveImage,
    "RandomDataset" : RandomDataset
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
#感觉就是别名
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example3": "我算什么东西？",
    "Load Image" : "Load Image",
    "Save Image" : "Save Image",
    "RandomDataset" : "RandomDataset"
    
}
