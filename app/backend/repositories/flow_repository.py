from typing import List, Optional
from sqlalchemy.orm import Session
from app.backend.database.models import HedgeFundFlow


class FlowRepository:
    """Repository for HedgeFundFlow CRUD operations, scoped by device_id"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_flow(self, device_id: str, name: str, nodes: dict, edges: dict, description: str = None,
                   viewport: dict = None, data: dict = None, is_template: bool = False, tags: List[str] = None) -> HedgeFundFlow:
        """Create a new hedge fund flow, owned by device_id"""
        flow = HedgeFundFlow(
            device_id=device_id,
            name=name,
            description=description,
            nodes=nodes,
            edges=edges,
            viewport=viewport,
            data=data,
            is_template=is_template,
            tags=tags or []
        )
        self.db.add(flow)
        self.db.commit()
        self.db.refresh(flow)
        return flow
    
    def get_flow_by_id(self, flow_id: int, device_id: str) -> Optional[HedgeFundFlow]:
        """Get a flow by its ID, only if it belongs to device_id"""
        return self.db.query(HedgeFundFlow).filter(
            HedgeFundFlow.id == flow_id,
            HedgeFundFlow.device_id == device_id,
        ).first()
    
    def get_all_flows(self, device_id: str, include_templates: bool = True) -> List[HedgeFundFlow]:
        """Get all flows belonging to device_id, optionally excluding templates"""
        query = self.db.query(HedgeFundFlow).filter(HedgeFundFlow.device_id == device_id)
        if not include_templates:
            query = query.filter(HedgeFundFlow.is_template == False)
        return query.order_by(HedgeFundFlow.updated_at.desc()).all()
    
    def get_flows_by_name(self, name: str, device_id: str) -> List[HedgeFundFlow]:
        """Search this device's flows by name (case-insensitive partial match)"""
        return self.db.query(HedgeFundFlow).filter(
            HedgeFundFlow.device_id == device_id,
            HedgeFundFlow.name.ilike(f"%{name}%"),
        ).order_by(HedgeFundFlow.updated_at.desc()).all()
    
    def update_flow(self, flow_id: int, device_id: str, name: str = None, description: str = None,
                   nodes: dict = None, edges: dict = None, viewport: dict = None, data: dict = None,
                   is_template: bool = None, tags: List[str] = None) -> Optional[HedgeFundFlow]:
        """Update an existing flow — only if it belongs to device_id"""
        flow = self.get_flow_by_id(flow_id, device_id)
        if not flow:
            return None
        
        if name is not None:
            flow.name = name
        if description is not None:
            flow.description = description
        if nodes is not None:
            flow.nodes = nodes
        if edges is not None:
            flow.edges = edges
        if viewport is not None:
            flow.viewport = viewport
        if data is not None:
            flow.data = data
        if is_template is not None:
            flow.is_template = is_template
        if tags is not None:
            flow.tags = tags
        
        self.db.commit()
        self.db.refresh(flow)
        return flow
    
    def delete_flow(self, flow_id: int, device_id: str) -> bool:
        """Delete a flow by ID — only if it belongs to device_id"""
        flow = self.get_flow_by_id(flow_id, device_id)
        if not flow:
            return False
        
        self.db.delete(flow)
        self.db.commit()
        return True
    
    def duplicate_flow(self, flow_id: int, device_id: str, new_name: str = None) -> Optional[HedgeFundFlow]:
        """Create a copy of an existing flow — only if the original belongs to device_id"""
        original = self.get_flow_by_id(flow_id, device_id)
        if not original:
            return None
        
        copy_name = new_name or f"{original.name} (Copy)"
        
        return self.create_flow(
            device_id=device_id,
            name=copy_name,
            description=original.description,
            nodes=original.nodes,
            edges=original.edges,
            viewport=original.viewport,
            data=original.data,
            is_template=False,  # Copies are not templates by default
            tags=original.tags
        )