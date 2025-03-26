// src/components/PaperList.js
import React from 'react';
import PaperItem from './PaperItem';

const PaperList = ({ papers }) => {
  return (
    <div className="paper-list">
      {papers.map((paper, index) => (
        <PaperItem
          key={index}
          title={paper.title}
          abstract={paper.abstract}
          distance={paper.distance}
        />
      ))}
    </div>
  );
};

export default PaperList;
