// src/components/PaperItem.js
import React from 'react';

const PaperItem = ({ title, abstract, distance }) => {
  return (
    <div className="paper-item">
      <h3>{title}</h3>
      <p>{abstract}</p>
      <p><strong>Distance:</strong> {distance.toFixed(3)}</p>
    </div>
  );
};

export default PaperItem;
