# -*- coding: utf-8 -*-
"""
fem3d — управляющий слой программного комплекса МКЭ для трёхмерной
теплопроводности. Связан с C++ ядром fem_core через ctypes (модуль
core_bridge).

Структура пакета:
    core_bridge — граница с ядром (12 функций C-API через ctypes);
    mesh        — пресеты, справочник материалов, импорт сеток;
    problem     — высокоуровневое описание задачи и оркестрация расчёта;
    postprocess — экспорт VTU/CSV/отчёт;
    verify      — верификационные задачи T1–T4.
"""

from .core_bridge import (
    BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN, BC_RADIATION,
    STEFAN_BOLTZMANN,
    CoreBridge, CoreError,
    FACE_NAMES, FACE_X_MINUS, FACE_X_PLUS,
    FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS,
    SolverInfo,
    VOLSRC_BOX, VOLSRC_SPHERE,
)
from .mesh import (
    BoxPreset, Material, MeshInfo, ShapePreset,
    MATERIALS, PRESETS, SHAPE_PRESETS,
    SUPPORTED_IMPORT_EXTENSIONS,
    all_materials, compute_mesh_info, import_mesh_file, import_msh,
    import_stl, import_step, import_vtu, load_user_materials,
    material_by_name, save_user_materials,
)
from .problem import (
    BoundaryCondition, BoxGeometry, CONVECTION_PRESETS, MaterialRegion,
    PointSource, Problem,
    REGION_BOX, REGION_SPHERE, VolumeSource,
    HEATING_TEMPLATES,
    HEATING_TEMPLATES_FULL,
    template_all_convection, template_bottom_heat_top_cool,
    template_cpu_cooler, template_heat_flux_bottom,
    template_hot_cold_walls, template_insulated_box,
    template_one_side_furnace, template_reset, template_solar_panel,
    template_top_heat_bottom_cool, template_water_cooled,
)
from .project import (
    PROJECT_EXTENSION, PROJECT_FORMAT_VERSION,
    save_project, load_project, project_info,
)
from .convection import (
    FluidProperties, ForcedConvectionResult,
    SHAPE_PLATE, SHAPE_CYLINDER, SHAPE_SPHERE, SHAPE_CUBE, SHAPE_NAMES,
    air_properties, reynolds, nusselt_forced, nusselt_natural_vertical_plate,
    rayleigh, heat_transfer_coefficient,
    surface_areas, frontal_area, faces_exposed_to_flow, parse_direction,
    characteristic_length, analyze_forced_convection,
    apply_forced_convection_bc,
    analyze_problem_air_flow, apply_problem_air_flow,
    convection_summary_text,
)

__version__ = "1.1.0"

__all__ = [
    # core_bridge
    "BC_DIRICHLET", "BC_NEUMANN", "BC_NONE", "BC_ROBIN", "BC_RADIATION",
    "STEFAN_BOLTZMANN",
    "CoreBridge", "CoreError",
    "FACE_NAMES", "FACE_X_MINUS", "FACE_X_PLUS",
    "FACE_Y_MINUS", "FACE_Y_PLUS", "FACE_Z_MINUS", "FACE_Z_PLUS",
    "SolverInfo",
    "VOLSRC_BOX", "VOLSRC_SPHERE",
    # mesh
    "BoxPreset", "Material", "MeshInfo", "ShapePreset",
    "MATERIALS", "PRESETS", "SHAPE_PRESETS",
    "SUPPORTED_IMPORT_EXTENSIONS",
    "all_materials", "compute_mesh_info", "import_mesh_file", "import_msh",
    "import_stl", "import_step", "import_vtu", "load_user_materials",
    "material_by_name", "save_user_materials",
    # problem
    "BoundaryCondition", "BoxGeometry", "CONVECTION_PRESETS",
    "MaterialRegion", "PointSource",
    "Problem", "REGION_BOX", "REGION_SPHERE", "VolumeSource",
    "HEATING_TEMPLATES",
    "HEATING_TEMPLATES_FULL",
    "template_all_convection", "template_bottom_heat_top_cool",
    "template_cpu_cooler", "template_heat_flux_bottom",
    "template_hot_cold_walls", "template_insulated_box",
    "template_one_side_furnace", "template_reset", "template_solar_panel",
    "template_top_heat_bottom_cool", "template_water_cooled",
    # project
    "PROJECT_EXTENSION", "PROJECT_FORMAT_VERSION",
    "save_project", "load_project", "project_info",
    # convection (вынужденная конвекция при обтекании)
    "FluidProperties", "ForcedConvectionResult",
    "SHAPE_PLATE", "SHAPE_CYLINDER", "SHAPE_SPHERE", "SHAPE_CUBE", "SHAPE_NAMES",
    "air_properties", "reynolds", "nusselt_forced",
    "nusselt_natural_vertical_plate", "rayleigh", "heat_transfer_coefficient",
    "surface_areas", "frontal_area", "faces_exposed_to_flow", "parse_direction",
    "characteristic_length", "analyze_forced_convection",
    "apply_forced_convection_bc",
    "analyze_problem_air_flow", "apply_problem_air_flow",
    "convection_summary_text",
]
