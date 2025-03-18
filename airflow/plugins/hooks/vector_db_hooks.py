from typing import Dict, List, Any, Optional
import os
import json
from airflow.hooks.base import BaseHook
from airflow.exceptions import AirflowException
import numpy as np

class PineconeHook(BaseHook):
    """
    Hook for Pinecone vector database
    """
    
    conn_name_attr = 'pinecone_conn_id'
    default_conn_name = 'pinecone_default'
    conn_type = 'pinecone'
    hook_name = 'Pinecone'
    
    def __init__(
        self, 
        pinecone_conn_id: str = default_conn_name,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None
    ):
        super().__init__()
        self.pinecone_conn_id = pinecone_conn_id
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self._pinecone = None
        self._index = None
    
    def get_conn(self):
        """
        Get Pinecone connection
        """
        if self._pinecone is not None:
            return self._pinecone
        
        try:
            import pinecone
            
            # Get connection details
            if not self.api_key:
                conn = self.get_connection(self.pinecone_conn_id)
                self.api_key = conn.password
                self.environment = conn.host
                self.index_name = conn.schema
            
            # Initialize Pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            self._pinecone = pinecone
            return self._pinecone
            
        except ImportError:
            raise AirflowException("Pinecone package not installed")
        except Exception as e:
            raise AirflowException(f"Failed to connect to Pinecone: {str(e)}")
    
    def get_index(self, index_name: Optional[str] = None):
        """
        Get Pinecone index
        """
        if self._index is not None:
            return self._index
        
        pinecone = self.get_conn()
        index_name = index_name or self.index_name
        
        if not index_name:
            raise AirflowException("Index name not provided")
        
        try:
            # Check if index exists
            if index_name not in pinecone.list_indexes():
                raise AirflowException(f"Index {index_name} does not exist")
            
            self._index = pinecone.Index(index_name)
            return self._index
            
        except Exception as e:
            raise AirflowException(f"Failed to get Pinecone index: {str(e)}")
    
    def upsert(
        self, 
        vectors: List[Dict[str, Any]], 
        index_name: Optional[str] = None
    ):
        """
        Upsert vectors into Pinecone index
        
        Args:
            vectors: List of dictionaries with 'id', 'values', and 'metadata'
            index_name: Optional index name
        """
        index = self.get_index(index_name)
        
        try:
            response = index.upsert(vectors=vectors)
            return response
        except Exception as e:
            raise AirflowException(f"Failed to upsert vectors: {str(e)}")
    
    def query(
        self, 
        vector: List[float], 
        top_k: int = 5, 
        filter: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None
    ):
        """
        Query vectors from Pinecone index
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Metadata filter
            index_name: Optional index name
        """
        index = self.get_index(index_name)
        
        try:
            response = index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            return response
        except Exception as e:
            raise AirflowException(f"Failed to query vectors: {str(e)}")


class ChromaDBHook(BaseHook):
    """
    Hook for ChromaDB vector database
    """
    
    conn_name_attr = 'chromadb_conn_id'
    default_conn_name = 'chromadb_default'
    conn_type = 'chromadb'
    hook_name = 'ChromaDB'
    
    def __init__(
        self, 
        chromadb_conn_id: str = default_conn_name,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        super().__init__()
        self.chromadb_conn_id = chromadb_conn_id
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    def get_conn(self):
        """
        Get ChromaDB connection
        """
        if self._client is not None:
            return self._client
        
        try:
            import chromadb
            
            # Get connection details
            if not self.persist_directory:
                conn = self.get_connection(self.chromadb_conn_id)
                self.persist_directory = conn.host
                self.collection_name = conn.schema
            
            # Initialize ChromaDB
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            return self._client
            
        except ImportError:
            raise AirflowException("ChromaDB package not installed")
        except Exception as e:
            raise AirflowException(f"Failed to connect to ChromaDB: {str(e)}")
    
    def get_collection(self, collection_name: Optional[str] = None):
        """
        Get ChromaDB collection
        """
        if self._collection is not None:
            return self._collection
        
        client = self.get_conn()
        collection_name = collection_name or self.collection_name
        
        if not collection_name:
            raise AirflowException("Collection name not provided")
        
        try:
            # Get or create collection
            self._collection = client.get_or_create_collection(name=collection_name)
            return self._collection
            
        except Exception as e:
            raise AirflowException(f"Failed to get ChromaDB collection: {str(e)}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ):
        """
        Add documents and embeddings to ChromaDB collection
        
        Args:
            ids: List of unique IDs
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
            documents: Optional list of document texts
            collection_name: Optional collection name
        """
        collection = self.get_collection(collection_name)
        
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            return {"success": True, "count": len(ids)}
        except Exception as e:
            raise AirflowException(f"Failed to add to ChromaDB: {str(e)}")
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ):
        """
        Query ChromaDB collection
        
        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            where: Optional metadata filter
            collection_name: Optional collection name
        """
        collection = self.get_collection(collection_name)
        
        try:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            raise AirflowException(f"Failed to query ChromaDB: {str(e)}")