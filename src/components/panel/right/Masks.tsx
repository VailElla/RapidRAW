import {
  Brush,
  BringToFront,
  Circle,
  Cloud,
  Droplet,
  Droplets,
  Eraser,
  MoreHorizontal,
  RectangleHorizontal,
  Sparkles,
  TriangleRight,
  User,
  Sun,
  Stamp,
  Bandage,
} from 'lucide-react';
import i18n from 'i18next';

export enum Mask {
  AiDepth = 'ai-depth',
  AiForeground = 'ai-foreground',
  AiSky = 'ai-sky',
  AiSubject = 'ai-subject',
  All = 'all',
  Brush = 'brush',
  Flow = 'flow',
  Color = 'color',
  Linear = 'linear',
  Luminance = 'luminance',
  QuickEraser = 'quick-eraser',
  Radial = 'radial',
  Clone = 'clone',
  Heal = 'heal',
}

export enum SubMaskMode {
  Additive = 'additive',
  Subtractive = 'subtractive',
  Intersect = 'intersect',
}

export enum ToolType {
  AiSeletor = 'ai-selector',
  Brush = 'brush',
  Eraser = 'eraser',
  GenerativeReplace = 'generative-replace',
  SelectSubject = 'select-subject',
}

export interface MaskType {
  disabled: boolean;
  icon: any;
  id?: string;
  name: string;
  type: Mask;
}

export interface SubMask {
  id: string;
  invert: boolean;
  mode: SubMaskMode;
  name?: string;
  opacity: number;
  parameters?: any;
  type: Mask;
  visible: boolean;
}

export function formatMaskTypeName(type: string) {
  if (type === Mask.AiDepth) return i18n.t('masks.types.depth', 'Depth');
  if (type === Mask.AiSubject) return i18n.t('masks.types.subject', 'Subject');
  if (type === Mask.AiForeground) return i18n.t('masks.types.foreground', 'Foreground');
  if (type === Mask.AiSky) return i18n.t('masks.types.sky', 'Sky');
  if (type === Mask.All) return i18n.t('masks.types.all', 'Whole Image');
  if (type === Mask.QuickEraser) return i18n.t('masks.types.quickEraser', 'Quick Erase');
  if (type === Mask.Brush) return i18n.t('masks.types.brush', 'Brush');
  if (type === Mask.Flow) return i18n.t('masks.types.flow', 'Flow');
  if (type === Mask.Color) return i18n.t('masks.types.color', 'Color');
  if (type === Mask.Linear) return i18n.t('masks.types.linear', 'Linear');
  if (type === Mask.Luminance) return i18n.t('masks.types.luminance', 'Luminance');
  if (type === Mask.Radial) return i18n.t('masks.types.radial', 'Radial');
  if (type === Mask.Clone) return i18n.t('masks.types.clone', 'Clone');
  if (type === Mask.Heal) return i18n.t('masks.types.heal', 'Heal');
  return type.charAt(0).toUpperCase() + type.slice(1);
}

export function getMaskTypeName(mask: MaskType) {
  if (mask.id === 'others') return i18n.t('masks.types.others', 'Others');
  if (mask.type === Mask.QuickEraser && mask.name === 'Quick Erase') {
    return i18n.t('masks.types.quickErase', 'Quick Erase');
  }
  return formatMaskTypeName(mask.type);
}

export function getSubMaskName(subMask: Pick<SubMask, 'name' | 'type'>) {
  return subMask.name?.trim() || formatMaskTypeName(subMask.type);
}

export const MASK_ICON_MAP: Record<Mask, any> = {
  [Mask.AiDepth]: BringToFront,
  [Mask.AiForeground]: User,
  [Mask.AiSky]: Cloud,
  [Mask.AiSubject]: Sparkles,
  [Mask.All]: RectangleHorizontal,
  [Mask.Brush]: Brush,
  [Mask.Flow]: Droplets,
  [Mask.Color]: Droplet,
  [Mask.Linear]: TriangleRight,
  [Mask.Luminance]: Sparkles,
  [Mask.QuickEraser]: Eraser,
  [Mask.Radial]: Circle,
  [Mask.Clone]: Stamp,
  [Mask.Heal]: Bandage,
};

export const MASK_PANEL_CREATION_TYPES: Array<MaskType> = [
  {
    disabled: false,
    icon: Sparkles,
    name: 'Subject',
    type: Mask.AiSubject,
  },
  {
    disabled: false,
    icon: Cloud,
    name: 'Sky',
    type: Mask.AiSky,
  },
  {
    disabled: false,
    icon: User,
    name: 'Foreground',
    type: Mask.AiForeground,
  },
  {
    disabled: false,
    icon: TriangleRight,
    name: 'Linear',
    type: Mask.Linear,
  },
  {
    disabled: false,
    icon: Circle,
    name: 'Radial',
    type: Mask.Radial,
  },
  {
    disabled: false,
    icon: MoreHorizontal,
    id: 'others',
    name: 'Others',
    type: null as any,
  },
];

export const AI_MANUAL_CLEANUP_TYPES: Array<MaskType> = [
  {
    disabled: false,
    icon: Stamp,
    name: 'Clone',
    type: Mask.Clone,
  },
  {
    disabled: false,
    icon: Bandage,
    name: 'Heal',
    type: Mask.Heal,
  },
];

export const AI_GENERATIVE_CREATION_TYPES: Array<MaskType> = [
  {
    disabled: false,
    icon: Eraser,
    name: 'Quick Erase',
    type: Mask.QuickEraser,
  },
  {
    disabled: false,
    icon: Sparkles,
    name: 'Subject',
    type: Mask.AiSubject,
  },
  {
    disabled: false,
    icon: User,
    name: 'Foreground',
    type: Mask.AiForeground,
  },
  {
    disabled: false,
    icon: Brush,
    name: 'Brush',
    type: Mask.Brush,
  },
  {
    disabled: false,
    icon: TriangleRight,
    name: 'Linear',
    type: Mask.Linear,
  },
  {
    disabled: false,
    icon: Circle,
    name: 'Radial',
    type: Mask.Radial,
  },
];

export const SUB_MASK_COMPONENT_TYPES: Array<MaskType> = [
  {
    disabled: false,
    icon: Sparkles,
    name: 'Subject',
    type: Mask.AiSubject,
  },
  {
    disabled: false,
    icon: Cloud,
    name: 'Sky',
    type: Mask.AiSky,
  },
  {
    disabled: false,
    icon: User,
    name: 'Foreground',
    type: Mask.AiForeground,
  },
  {
    disabled: false,
    icon: TriangleRight,
    name: 'Linear',
    type: Mask.Linear,
  },
  {
    disabled: false,
    icon: Circle,
    name: 'Radial',
    type: Mask.Radial,
  },
  {
    disabled: false,
    icon: MoreHorizontal,
    id: 'others',
    name: 'Others',
    type: null as any,
  },
];

export const OTHERS_MASK_TYPES: Array<MaskType> = [
  {
    disabled: false,
    icon: BringToFront,
    name: 'Depth',
    type: Mask.AiDepth,
  },
  {
    disabled: false,
    icon: Droplet,
    name: 'Color',
    type: Mask.Color,
  },
  {
    disabled: false,
    icon: Sun,
    name: 'Luminance',
    type: Mask.Luminance,
  },
  {
    disabled: false,
    icon: Brush,
    name: 'Brush',
    type: Mask.Brush,
  },
  {
    disabled: false,
    icon: Droplets,
    name: 'Flow',
    type: Mask.Flow,
  },
  {
    disabled: false,
    icon: RectangleHorizontal,
    name: 'Whole Image',
    type: Mask.All,
  },
];

export const AI_SUB_MASK_COMPONENT_TYPES: Array<MaskType> = [
  ...AI_MANUAL_CLEANUP_TYPES,
  ...AI_GENERATIVE_CREATION_TYPES,
];
