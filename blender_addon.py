# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

bl_info = {
    "name": "Hunyuan3D-2 Generator",
    "author": "Tencent Hunyuan3D",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Hunyuan3D-2 3D Generator",
    "description": "Generate/Texturing 3D models from text descriptions or images",
    "category": "3D View",
}
import base64
import os
import tempfile
import threading

import bpy
import requests
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty


class Hunyuan3DProperties(bpy.types.PropertyGroup):
    prompt: StringProperty(
        name="Text Prompt",
        description="Describe what you want to generate",
        default=""
    )
    api_url: StringProperty(
        name="API URL",
        description="URL of the Text-to-3D API service",
        default="http://localhost:8081"
    )
    is_processing: BoolProperty(
        name="Processing",
        default=False
    )
    job_id: StringProperty(
        name="Job ID",
        default=""
    )
    status_message: StringProperty(
        name="Status Message",
        default=""
    )
    # 添加图片路径属性
    image_path: StringProperty(
        name="Front View",
        description="Select the front view image",
        subtype='FILE_PATH'
    )
    image_left_path: StringProperty(
        name="Left View",
        description="Select the left view image",
        subtype='FILE_PATH'
    )
    image_back_path: StringProperty(
        name="Back View",
        description="Select the back view image",
        subtype='FILE_PATH'
    )
    image_right_path: StringProperty(
        name="Right View",
        description="Select the right view image",
        subtype='FILE_PATH'
    )
    # 修改后的 octree_resolution 属性
    octree_resolution: IntProperty(
        name="Octree Resolution",
        description="Octree resolution for the 3D generation",
        default=256,
        min=128,
        max=512,
    )
    num_inference_steps: IntProperty(
        name="Number of Inference Steps",
        description="Number of inference steps for the 3D generation",
        default=20,
        min=20,
        max=50
    )
    guidance_scale: FloatProperty(
        name="Guidance Scale",
        description="Guidance scale for the 3D generation",
        default=5.5,
        min=1.0,
        max=10.0
    )
    # 添加 texture 属性
    texture: BoolProperty(
        name="Generate Texture",
        description="Whether to generate texture for the 3D model",
        default=False
    )


class Hunyuan3DOperator(bpy.types.Operator):
    bl_idname = "object.generate_3d"
    bl_label = "Generate 3D Model"
    bl_description = "Generate a 3D model from text description, an image or a selected mesh"

    job_id = ''
    prompt = ""
    api_url = ""
    image_path = ""
    image_left_path = ""
    image_back_path = ""
    image_right_path = ""
    octree_resolution = 256
    num_inference_steps = 20
    guidance_scale = 5.5
    texture = False  # 新增属性
    selected_mesh_base64 = ""
    selected_mesh = None  # 新增属性，用于存储选中的 mesh

    thread = None
    task_finished = False

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return {'CANCELLED'}

        if self.task_finished:
            print("Threaded task completed")
            self.task_finished = False
            props = context.scene.gen_3d_props
            props.is_processing = False

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        # 启动线程
        props = context.scene.gen_3d_props
        self.prompt = props.prompt
        self.api_url = props.api_url
        self.image_path = props.image_path
        self.image_left_path = props.image_left_path
        self.image_back_path = props.image_back_path
        self.image_right_path = props.image_right_path
        self.octree_resolution = props.octree_resolution
        self.num_inference_steps = props.num_inference_steps
        self.guidance_scale = props.guidance_scale
        self.texture = props.texture  # 获取 texture 属性的值

        if self.prompt == "" and self.image_path == "":
            self.report({'WARNING'}, "Please enter some text or select an image first.")
            return {'FINISHED'}

        # 保存选中的 mesh 对象引用
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                self.selected_mesh = obj
                break

        if self.selected_mesh:
            temp_glb_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            temp_glb_file.close()
            bpy.ops.export_scene.gltf(filepath=temp_glb_file.name, use_selection=True)
            with open(temp_glb_file.name, "rb") as file:
                mesh_data = file.read()
            mesh_b64_str = base64.b64encode(mesh_data).decode()
            os.unlink(temp_glb_file.name)
            self.selected_mesh_base64 = mesh_b64_str

        props.is_processing = True

        # 将相对路径转换为相对于 Blender 文件所在目录的绝对路径
        blend_file_dir = os.path.dirname(bpy.data.filepath)
        self.report({'INFO'}, f"blend_file_dir {blend_file_dir}")
        self.report({'INFO'}, f"image_path {self.image_path}")
        if self.image_path.startswith('//'):
            self.image_path = self.image_path[2:]
            self.image_path = os.path.join(blend_file_dir, self.image_path)

        if self.selected_mesh and self.texture:
            props.status_message = "Texturing Selected Mesh...\n" \
                                   "This may take several minutes depending \n on your GPU power."
        else:
            mesh_type = 'Textured Mesh' if self.texture else 'White Mesh'
            prompt_type = 'Text Prompt' if self.prompt else 'Image'
            props.status_message = f"Generating {mesh_type} with {prompt_type}...\n" \
                                   "This may take several minutes depending \n on your GPU power."

        self.thread = threading.Thread(target=self.generate_model, args=[context])
        self.thread.start()

        wm = context.window_manager
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def generate_model(self, context):
        self.report({'INFO'}, f"Generation Start")
        base_url = self.api_url.rstrip('/')

        try:
            if self.selected_mesh_base64 and self.texture:
                # Texturing the selected mesh
                if self.image_path and os.path.exists(self.image_path):
                    self.report({'INFO'}, f"Post Texturing with Image")
                    # 打开图片文件并以二进制模式读取
                    with open(self.image_path, "rb") as file:
                        # 读取文件内容
                        image_data = file.read()
                    # 对图片数据进行 Base64 编码
                    img_b64_str = base64.b64encode(image_data).decode()
                    payload = {
                        "mesh": self.selected_mesh_base64,
                        "image": img_b64_str,
                        "octree_resolution": self.octree_resolution,
                        "num_inference_steps": self.num_inference_steps,
                        "guidance_scale": self.guidance_scale,
                        "texture": self.texture
                    }
                    extra_views = [
                        ('image_left_path', 'image_left'),
                        ('image_back_path', 'image_back'),
                        ('image_right_path', 'image_right'),
                    ]
                    for attr, key in extra_views:
                        path = getattr(self, attr, '')
                        if path and os.path.exists(path):
                            with open(path, "rb") as f:
                                payload[key] = base64.b64encode(f.read()).decode()
                    response = requests.post(
                        f"{base_url}/generate",
                        json=payload,
                    )
                else:
                    self.report({'INFO'}, f"Post Texturing with Text")
                    response = requests.post(
                        f"{base_url}/generate",
                        json={
                            "mesh": self.selected_mesh_base64,
                            "text": self.prompt,
                            "octree_resolution": self.octree_resolution,
                            "num_inference_steps": self.num_inference_steps,
                            "guidance_scale": self.guidance_scale,
                            "texture": self.texture  # 传递 texture 参数
                        },
                    )
            else:
                if self.image_path:
                    if not os.path.exists(self.image_path):
                        self.report({'ERROR'}, f"Image path does not exist {self.image_path}")
                        raise Exception(f'Image path does not exist {self.image_path}')
                    self.report({'INFO'}, f"Post Start Image to 3D")
                    # 打开图片文件并以二进制模式读取
                    with open(self.image_path, "rb") as file:
                        # 读取文件内容
                        image_data = file.read()
                    # 对图片数据进行 Base64 编码
                    img_b64_str = base64.b64encode(image_data).decode()
                    payload = {
                        "image": img_b64_str,
                        "octree_resolution": self.octree_resolution,
                        "num_inference_steps": self.num_inference_steps,
                        "guidance_scale": self.guidance_scale,
                        "texture": self.texture
                    }
                    extra_views = [
                        ('image_left_path', 'image_left'),
                        ('image_back_path', 'image_back'),
                        ('image_right_path', 'image_right'),
                    ]
                    for attr, key in extra_views:
                        path = getattr(self, attr, '')
                        if path and os.path.exists(path):
                            with open(path, "rb") as f:
                                payload[key] = base64.b64encode(f.read()).decode()
                    response = requests.post(
                        f"{base_url}/generate",
                        json=payload,
                    )
                else:
                    self.report({'INFO'}, f"Post Start Text to 3D")
                    response = requests.post(
                        f"{base_url}/generate",
                        json={
                            "text": self.prompt,
                            "octree_resolution": self.octree_resolution,
                            "num_inference_steps": self.num_inference_steps,
                            "guidance_scale": self.guidance_scale,
                            "texture": self.texture  # 传递 texture 参数
                        },
                    )
            self.report({'INFO'}, f"Post Done")
            self.task_finished = True
            props = context.scene.gen_3d_props
            props.is_processing = False

            if response.status_code != 200:
                self.report({'ERROR'}, f"Generation failed: {response.text}")
                return

            # Decode base64 and save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            temp_file.write(response.content)
            temp_file.close()

            # Import the GLB file in the main thread
            def import_handler():
                bpy.ops.import_scene.gltf(filepath=temp_file.name)
                os.unlink(temp_file.name)

                # 获取新导入的 mesh
                new_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
                if new_obj and self.selected_mesh and self.texture:
                    # 应用选中 mesh 的位置、旋转和缩放
                    new_obj.location = self.selected_mesh.location
                    new_obj.rotation_euler = self.selected_mesh.rotation_euler
                    new_obj.scale = self.selected_mesh.scale

                    # 隐藏原来的 mesh
                    self.selected_mesh.hide_set(True)
                    self.selected_mesh.hide_render = True

                return None

            bpy.app.timers.register(import_handler)

        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")

        finally:
            self.task_finished = True
            props = context.scene.gen_3d_props
            props.is_processing = False
            self.selected_mesh_base64 = ""


class Hunyuan3DPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hunyuan3D-2'
    bl_label = 'Hunyuan3D-2 3D Generator'

    def draw(self, context):
        layout = self.layout
        props = context.scene.gen_3d_props

        layout.prop(props, "api_url")
        layout.prop(props, "prompt")
        # 添加图片选择器
        layout.prop(props, "image_path")
        layout.prop(props, "image_left_path")
        layout.prop(props, "image_back_path")
        layout.prop(props, "image_right_path")
        # 添加新属性的 UI 元素
        layout.prop(props, "octree_resolution")
        layout.prop(props, "num_inference_steps")
        layout.prop(props, "guidance_scale")
        # 添加 texture 属性的 UI 元素
        layout.prop(props, "texture")

        row = layout.row()
        row.enabled = not props.is_processing
        row.operator("object.generate_3d")

        if props.is_processing:
            if props.status_message:
                for line in props.status_message.split("\n"):
                    layout.label(text=line)
            else:
                layout.label("Processing...")


classes = (
    Hunyuan3DProperties,
    Hunyuan3DOperator,
    Hunyuan3DPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gen_3d_props = bpy.props.PointerProperty(type=Hunyuan3DProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gen_3d_props


if __name__ == "__main__":
    register()
